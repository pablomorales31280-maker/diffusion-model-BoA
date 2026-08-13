from __future__ import annotations

import argparse
import csv
import logging
import math
import random
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import core.logger as Logger
import data as Data
import model as Model


def unwrap(module):
    if isinstance(module, nn.DataParallel):
        return module.module
    return module


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def center_crop_batch(
    tensor: torch.Tensor,
    crop_size: int,
) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(
            f"Expected a BCHW tensor, got shape {tuple(tensor.shape)}."
        )

    height, width = tensor.shape[-2:]

    if height < crop_size or width < crop_size:
        raise ValueError(
            f"Input size {height}x{width} is smaller than "
            f"the requested crop size {crop_size}x{crop_size}."
        )

    top = (height - crop_size) // 2
    left = (width - crop_size) // 2

    return tensor[
        ...,
        top : top + crop_size,
        left : left + crop_size,
    ]


def prepare_test_batch(
    batch: dict,
    crop_size: int | None,
    crop_mode: str,
) -> dict:
    result = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.clone()
        else:
            result[key] = value

    if crop_size is None:
        return result

    if crop_mode != "center":
        raise ValueError(
            f"Unsupported test crop mode: {crop_mode}. "
            "Currently supported: center."
        )

    for key in ("SR", "HR", "LR"):
        if key in result and torch.is_tensor(result[key]):
            result[key] = center_crop_batch(
                result[key],
                crop_size,
            )

    return result


def tensor_to_rgb_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().float().cpu()

    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(
                "Image saving currently expects batch size 1."
            )
        tensor = tensor[0]

    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(
            f"Expected RGB CHW tensor, got shape {tuple(tensor.shape)}."
        )

    tensor = tensor.clamp(0.0, 1.0)

    image = (
        tensor.permute(1, 2, 0)
        .contiguous()
        .numpy()
    )

    return image


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    image = tensor_to_rgb_numpy(tensor)
    return np.round(image * 255.0).astype(np.uint8)


def calculate_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    prediction = prediction.detach().float().clamp(0.0, 1.0)
    target = target.detach().float().clamp(0.0, 1.0)

    mse = F.mse_loss(
        prediction,
        target,
        reduction="mean",
    ).item()

    if mse == 0.0:
        return float("inf")

    return 10.0 * math.log10(1.0 / mse)


def _ssim_single_channel(
    image_a: np.ndarray,
    image_b: np.ndarray,
) -> float:
    image_a = image_a.astype(np.float64)
    image_b = image_b.astype(np.float64)

    c1 = 0.01**2
    c2 = 0.03**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu_a = cv2.filter2D(
        image_a,
        -1,
        window,
    )[5:-5, 5:-5]

    mu_b = cv2.filter2D(
        image_b,
        -1,
        window,
    )[5:-5, 5:-5]

    mu_a_sq = mu_a**2
    mu_b_sq = mu_b**2
    mu_ab = mu_a * mu_b

    sigma_a_sq = (
        cv2.filter2D(
            image_a**2,
            -1,
            window,
        )[5:-5, 5:-5]
        - mu_a_sq
    )

    sigma_b_sq = (
        cv2.filter2D(
            image_b**2,
            -1,
            window,
        )[5:-5, 5:-5]
        - mu_b_sq
    )

    sigma_ab = (
        cv2.filter2D(
            image_a * image_b,
            -1,
            window,
        )[5:-5, 5:-5]
        - mu_ab
    )

    numerator = (
        (2.0 * mu_ab + c1)
        * (2.0 * sigma_ab + c2)
    )

    denominator = (
        (mu_a_sq + mu_b_sq + c1)
        * (sigma_a_sq + sigma_b_sq + c2)
    )

    return float(
        np.mean(numerator / denominator)
    )


def calculate_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    prediction_np = tensor_to_rgb_numpy(prediction)
    target_np = tensor_to_rgb_numpy(target)

    values = [
        _ssim_single_channel(
            prediction_np[:, :, channel],
            target_np[:, :, channel],
        )
        for channel in range(3)
    ]

    return float(np.mean(values))


def attack_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
) -> torch.Tensor:
    loss_name = loss_name.lower()

    if loss_name == "l1":
        return F.l1_loss(
            prediction,
            target,
            reduction="mean",
        )

    if loss_name in ("l2", "mse"):
        return F.mse_loss(
            prediction,
            target,
            reduction="mean",
        )

    raise ValueError(
        f"Unsupported PGD loss: {loss_name}. "
        "Supported losses: l1, l2, mse."
    )


def pgd_attack_prenet(
    diffusion,
    clean_input: torch.Tensor,
    target: torch.Tensor,
    attack_opt: dict,
) -> torch.Tensor:
    prenet = unwrap(diffusion.netP)

    epsilon = float(
        attack_opt["epsilon_255"]
    ) / 255.0

    step_size = float(
        attack_opt["step_size_255"]
    ) / 255.0

    steps = int(attack_opt["steps"])
    random_start = bool(
        attack_opt.get("random_start", True)
    )
    loss_name = attack_opt.get("loss", "l1")

    clip_min = float(
        attack_opt.get("clip_min", 0.0)
    )
    clip_max = float(
        attack_opt.get("clip_max", 1.0)
    )

    if epsilon < 0:
        raise ValueError(
            "epsilon_255 must be non-negative."
        )

    if step_size <= 0:
        raise ValueError(
            "step_size_255 must be positive."
        )

    if steps <= 0:
        raise ValueError(
            "PGD steps must be positive."
        )

    clean_input = clean_input.detach()
    target = target.detach()

    was_training = prenet.training
    prenet.eval()

    if random_start:
        adversarial = clean_input + torch.empty_like(
            clean_input
        ).uniform_(
            -epsilon,
            epsilon,
        )
    else:
        adversarial = clean_input.clone()

    adversarial = adversarial.clamp(
        clip_min,
        clip_max,
    )

    try:
        for _ in range(steps):
            adversarial = (
                adversarial.detach()
                .requires_grad_(True)
            )

            prediction = prenet(
                adversarial,
                time=None,
            )

            loss = attack_loss(
                prediction,
                target,
                loss_name,
            )

            gradient = torch.autograd.grad(
                loss,
                adversarial,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0]

            adversarial = (
                adversarial.detach()
                + step_size * gradient.sign()
            )

            perturbation = (
                adversarial - clean_input
            ).clamp(
                -epsilon,
                epsilon,
            )

            adversarial = (
                clean_input + perturbation
            ).clamp(
                clip_min,
                clip_max,
            )

    finally:
        prenet.train(was_training)

    return adversarial.detach()


def pgd_attack_joint_diffusion_loss(
    diffusion,
    clean_input: torch.Tensor,
    target: torch.Tensor,
    attack_opt: dict,
) -> torch.Tensor:
    """
    Diffusion-aware PGD surrogate.

    This attack maximizes the denoising training objective at one fixed
    diffusion timestep. It avoids differentiating through the complete
    stochastic reverse diffusion chain, which would be prohibitively
    expensive for iterative PGD.
    """
    prenet = unwrap(diffusion.netP)
    gaussian = unwrap(diffusion.netG)
    denoiser = gaussian.denoise_fn

    epsilon = float(
        attack_opt["epsilon_255"]
    ) / 255.0

    step_size = float(
        attack_opt["step_size_255"]
    ) / 255.0

    steps = int(attack_opt["steps"])
    random_start = bool(
        attack_opt.get("random_start", True)
    )
    loss_name = attack_opt.get("loss", "l1")

    clip_min = float(
        attack_opt.get("clip_min", 0.0)
    )
    clip_max = float(
        attack_opt.get("clip_max", 1.0)
    )

    timestep = attack_opt.get(
        "diffusion_timestep",
        None,
    )

    if timestep is None:
        timestep = gaussian.num_timesteps // 2

    timestep = int(timestep)

    if not (
        0 <= timestep < gaussian.num_timesteps
    ):
        raise ValueError(
            f"diffusion_timestep={timestep} is outside "
            f"[0, {gaussian.num_timesteps - 1}]."
        )

    clean_input = clean_input.detach()
    target = target.detach()

    batch_size = clean_input.shape[0]

    sqrt_alpha = (
        gaussian.sqrt_alphas_cumprod[timestep]
        .detach()
        .to(clean_input.device)
    )

    noise_level = sqrt_alpha.reshape(
        1,
        1,
    ).repeat(
        batch_size,
        1,
    )

    fixed_noise = torch.randn_like(
        clean_input
    )

    prenet_was_training = prenet.training
    gaussian_was_training = gaussian.training

    prenet.eval()
    gaussian.eval()

    if random_start:
        adversarial = clean_input + torch.empty_like(
            clean_input
        ).uniform_(
            -epsilon,
            epsilon,
        )
    else:
        adversarial = clean_input.clone()

    adversarial = adversarial.clamp(
        clip_min,
        clip_max,
    )

    try:
        for _ in range(steps):
            adversarial = (
                adversarial.detach()
                .requires_grad_(True)
            )

            initial_prediction = prenet(
                adversarial,
                time=None,
            )

            residual = (
                target - initial_prediction
            )

            noisy_residual = gaussian.q_sample(
                x_start=residual,
                continuous_sqrt_alpha_cumprod=(
                    noise_level.view(
                        -1,
                        1,
                        1,
                        1,
                    )
                ),
                noise=fixed_noise,
            )

            predicted_noise = denoiser(
                torch.cat(
                    [
                        adversarial,
                        noisy_residual,
                    ],
                    dim=1,
                ),
                noise_level,
            )

            loss = attack_loss(
                predicted_noise,
                fixed_noise,
                loss_name,
            )

            gradient = torch.autograd.grad(
                loss,
                adversarial,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0]

            adversarial = (
                adversarial.detach()
                + step_size * gradient.sign()
            )

            perturbation = (
                adversarial - clean_input
            ).clamp(
                -epsilon,
                epsilon,
            )

            adversarial = (
                clean_input + perturbation
            ).clamp(
                clip_min,
                clip_max,
            )

    finally:
        prenet.train(prenet_was_training)
        gaussian.train(gaussian_was_training)

    return adversarial.detach()


def build_adversarial_input(
    diffusion,
    clean_input: torch.Tensor,
    target: torch.Tensor,
    attack_opt: dict,
) -> torch.Tensor:
    attack_type = str(
        attack_opt.get("type", "pgd")
    ).lower()

    if attack_type != "pgd":
        raise ValueError(
            f"Unsupported attack type: {attack_type}. "
            "Currently supported: pgd."
        )

    norm = str(
        attack_opt.get("norm", "linf")
    ).lower()

    if norm not in (
        "linf",
        "l_inf",
        "infinity",
    ):
        raise ValueError(
            f"Unsupported PGD norm: {norm}. "
            "Currently supported: L-infinity."
        )

    target_name = str(
        attack_opt.get(
            "target",
            "prenet",
        )
    ).lower()

    if target_name == "prenet":
        return pgd_attack_prenet(
            diffusion,
            clean_input,
            target,
            attack_opt,
        )

    if target_name in (
        "joint_diffusion_loss",
        "diffusion_loss",
    ):
        return pgd_attack_joint_diffusion_loss(
            diffusion,
            clean_input,
            target,
            attack_opt,
        )

    raise ValueError(
        f"Unsupported PGD target: {target_name}. "
        "Supported targets: prenet, joint_diffusion_loss."
    )


def run_predict_and_refine(
    diffusion,
    input_image: torch.Tensor,
    target: torch.Tensor,
    sample_seed: int,
):
    data = {
        "SR": input_image,
        "HR": target,
        "Index": torch.zeros(
            input_image.shape[0],
            dtype=torch.long,
            device=input_image.device,
        ),
    }

    diffusion.feed_data(data)

    # Use the same reverse-diffusion randomness for clean and attacked
    # inference so that their comparison is not dominated by sampling noise.
    seed_everything(sample_seed)

    diffusion.test(
        continous=False
    )

    if not hasattr(diffusion, "IP"):
        raise RuntimeError(
            "The DDPM test() method did not expose the initial prediction "
            "as self.IP. Apply the predict-and-refine test() modification "
            "before using this test script."
        )

    if not hasattr(diffusion, "RS"):
        raise RuntimeError(
            "The DDPM test() method did not expose the generated residual "
            "as self.RS. Apply the predict-and-refine test() modification "
            "before using this test script."
        )

    if diffusion.SR.ndim != 4:
        raise RuntimeError(
            f"Final output has shape {tuple(diffusion.SR.shape)}. "
            "p_sample_loop() must return the final BCHW tensor `img`, "
            "not `ret_img[-1]`."
        )

    return {
        "input": input_image.detach().float().cpu(),
        "initial": diffusion.IP.detach().float().cpu(),
        "residual": diffusion.RS.detach().float().cpu(),
        "final": diffusion.SR.detach().float().cpu(),
        "target": target.detach().float().cpu(),
    }


def create_labeled_panel(
    image: np.ndarray,
    label: str,
    label_height: int = 26,
) -> Image.Image:
    pil_image = Image.fromarray(
        image,
        mode="RGB",
    )

    panel = Image.new(
        "RGB",
        (
            pil_image.width,
            pil_image.height + label_height,
        ),
        "white",
    )

    panel.paste(
        pil_image,
        (0, label_height),
    )

    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    draw.text(
        (6, 7),
        label,
        fill="black",
        font=font,
    )

    return panel


def save_comparison(
    output_path: Path,
    clean_result: dict,
    attacked_result: dict | None,
    adversarial_input: torch.Tensor | None,
) -> None:
    panels = [
        create_labeled_panel(
            tensor_to_uint8(
                clean_result["input"]
            ),
            "Blurry input",
        ),
        create_labeled_panel(
            tensor_to_uint8(
                clean_result["initial"]
            ),
            "Initial prediction",
        ),
        create_labeled_panel(
            tensor_to_uint8(
                clean_result["final"]
            ),
            "Clean restoration",
        ),
    ]

    if (
        attacked_result is not None
        and adversarial_input is not None
    ):
        panels.extend(
            [
                create_labeled_panel(
                    tensor_to_uint8(
                        adversarial_input
                    ),
                    "Adversarial input",
                ),
                create_labeled_panel(
                    tensor_to_uint8(
                        attacked_result[
                            "final"
                        ]
                    ),
                    "Attacked restoration",
                ),
            ]
        )

    panels.append(
        create_labeled_panel(
            tensor_to_uint8(
                clean_result["target"]
            ),
            "Ground truth",
        )
    )

    total_width = sum(
        panel.width
        for panel in panels
    )

    total_height = max(
        panel.height
        for panel in panels
    )

    canvas = Image.new(
        "RGB",
        (
            total_width,
            total_height,
        ),
        "white",
    )

    x_offset = 0

    for panel in panels:
        canvas.paste(
            panel,
            (x_offset, 0),
        )

        x_offset += panel.width

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(output_path)


def perturbation_metrics(
    clean: torch.Tensor,
    adversarial: torch.Tensor,
) -> tuple[float, float]:
    difference = (
        adversarial.detach().float()
        - clean.detach().float()
    )

    linf_255 = float(
        difference.abs().max().cpu()
        * 255.0
    )

    l2 = float(
        difference.reshape(
            difference.shape[0],
            -1,
        )
        .norm(
            p=2,
            dim=1,
        )
        .mean()
        .cpu()
    )

    return linf_255, l2


def mean_or_nan(values):
    if not values:
        return float("nan")
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the predict-and-refine diffusion model "
            "with optional PGD robustness testing."
        )
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the JSON test configuration.",
    )

    parser.add_argument(
        "-p",
        "--phase",
        type=str,
        choices=["val"],
        default="val",
    )

    parser.add_argument(
        "-gpu",
        "--gpu_ids",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-debug",
        "-d",
        action="store_true",
    )

    parser.add_argument(
        "-enable_wandb",
        action="store_true",
    )

    # These fields keep compatibility with core.logger.parse().
    parser.add_argument(
        "-log_wandb_ckpt",
        action="store_true",
    )

    parser.add_argument(
        "-log_eval",
        action="store_true",
    )

    args = parser.parse_args()

    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    Logger.setup_logger(
        None,
        opt["path"]["log"],
        "test",
        level=logging.INFO,
        screen=True,
    )

    logger = logging.getLogger("base")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    test_opt = opt.get("test", {})
    attack_opt = opt.get("attack", {})

    base_seed = int(
        test_opt.get("seed", 0)
    )

    deterministic = bool(
        test_opt.get(
            "deterministic",
            True,
        )
    )

    if deterministic:
        torch.backends.cudnn.deterministic = True

    seed_everything(base_seed)

    crop_size = test_opt.get(
        "crop_size",
        128,
    )

    if crop_size is not None:
        crop_size = int(crop_size)

    crop_mode = str(
        test_opt.get(
            "crop_mode",
            "center",
        )
    ).lower()

    save_images = bool(
        test_opt.get(
            "save_images",
            True,
        )
    )

    attack_enabled = bool(
        attack_opt.get(
            "enabled",
            False,
        )
    )

    logger.info("Starting model evaluation.")
    logger.info(
        "Checkpoint prefix: %s",
        opt["path"]["resume_state"],
    )
    logger.info(
        "Attack enabled: %s",
        attack_enabled,
    )

    if attack_enabled:
        logger.info(
            "Attack: %s | target=%s | epsilon=%.4f/255 | "
            "step_size=%.4f/255 | steps=%d",
            attack_opt.get("type", "pgd"),
            attack_opt.get("target", "prenet"),
            float(attack_opt["epsilon_255"]),
            float(attack_opt["step_size_255"]),
            int(attack_opt["steps"]),
        )

    validation_opt = opt["datasets"]["val"]

    validation_set = Data.create_dataset(
        validation_opt,
        "val",
    )

    validation_loader = Data.create_dataloader(
        validation_set,
        validation_opt,
        "val",
    )

    logger.info(
        "Validation dataset loaded with %d samples.",
        len(validation_set),
    )

    diffusion = Model.create_model(opt)

    diffusion.set_new_noise_schedule(
        opt["model"]["beta_schedule"]["val"],
        schedule_phase="val",
    )

    device = diffusion.device

    logger.info(
        "Model loaded on device: %s",
        device,
    )

    result_root = Path(
        opt["path"]["results"]
    )

    image_root = (
        result_root / "images"
    )

    result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if save_images:
        image_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    csv_path = (
        result_root / "metrics.csv"
    )

    rows = []

    clean_final_psnr_values = []
    clean_final_ssim_values = []
    attacked_final_psnr_values = []
    attacked_final_ssim_values = []
    psnr_drop_values = []

    for sample_index, batch in enumerate(
        validation_loader,
        start=1,
    ):
        batch = prepare_test_batch(
            batch,
            crop_size=crop_size,
            crop_mode=crop_mode,
        )

        clean_input = (
            batch["SR"]
            .to(device)
            .float()
        )

        target = (
            batch["HR"]
            .to(device)
            .float()
        )

        sample_seed = (
            base_seed + sample_index
        )

        clean_result = run_predict_and_refine(
            diffusion,
            clean_input,
            target,
            sample_seed=sample_seed,
        )

        clean_input_psnr = calculate_psnr(
            clean_result["input"],
            clean_result["target"],
        )

        clean_initial_psnr = calculate_psnr(
            clean_result["initial"],
            clean_result["target"],
        )

        clean_final_psnr = calculate_psnr(
            clean_result["final"],
            clean_result["target"],
        )

        clean_final_ssim = calculate_ssim(
            clean_result["final"],
            clean_result["target"],
        )

        clean_final_psnr_values.append(
            clean_final_psnr
        )

        clean_final_ssim_values.append(
            clean_final_ssim
        )

        adversarial_input = None
        attacked_result = None

        attacked_initial_psnr = float("nan")
        attacked_final_psnr = float("nan")
        attacked_final_ssim = float("nan")
        final_psnr_drop = float("nan")
        perturbation_linf_255 = 0.0
        perturbation_l2 = 0.0

        if attack_enabled:
            attack_seed = (
                base_seed
                + 100000
                + sample_index
            )

            seed_everything(
                attack_seed
            )

            adversarial_input = (
                build_adversarial_input(
                    diffusion,
                    clean_input,
                    target,
                    attack_opt,
                )
            )

            (
                perturbation_linf_255,
                perturbation_l2,
            ) = perturbation_metrics(
                clean_input,
                adversarial_input,
            )

            attacked_result = (
                run_predict_and_refine(
                    diffusion,
                    adversarial_input,
                    target,
                    sample_seed=sample_seed,
                )
            )

            attacked_initial_psnr = (
                calculate_psnr(
                    attacked_result["initial"],
                    attacked_result["target"],
                )
            )

            attacked_final_psnr = (
                calculate_psnr(
                    attacked_result["final"],
                    attacked_result["target"],
                )
            )

            attacked_final_ssim = (
                calculate_ssim(
                    attacked_result["final"],
                    attacked_result["target"],
                )
            )

            final_psnr_drop = (
                clean_final_psnr
                - attacked_final_psnr
            )

            attacked_final_psnr_values.append(
                attacked_final_psnr
            )

            attacked_final_ssim_values.append(
                attacked_final_ssim
            )

            psnr_drop_values.append(
                final_psnr_drop
            )

        row = OrderedDict(
            sample=sample_index,
            clean_input_psnr=clean_input_psnr,
            clean_initial_psnr=clean_initial_psnr,
            clean_final_psnr=clean_final_psnr,
            clean_final_ssim=clean_final_ssim,
            attack_enabled=attack_enabled,
            perturbation_linf_255=perturbation_linf_255,
            perturbation_l2=perturbation_l2,
            attacked_initial_psnr=attacked_initial_psnr,
            attacked_final_psnr=attacked_final_psnr,
            attacked_final_ssim=attacked_final_ssim,
            final_psnr_drop=final_psnr_drop,
        )

        rows.append(row)

        logger.info(
            "Sample %d | clean final PSNR=%.4f dB | "
            "clean final SSIM=%.6f",
            sample_index,
            clean_final_psnr,
            clean_final_ssim,
        )

        if attack_enabled:
            logger.info(
                "Sample %d | attacked final PSNR=%.4f dB | "
                "attacked final SSIM=%.6f | "
                "PSNR drop=%.4f dB | "
                "Linf=%.4f/255",
                sample_index,
                attacked_final_psnr,
                attacked_final_ssim,
                final_psnr_drop,
                perturbation_linf_255,
            )

        if save_images:
            comparison_path = (
                image_root
                / f"sample_{sample_index:04d}.png"
            )

            save_comparison(
                comparison_path,
                clean_result=clean_result,
                attacked_result=attacked_result,
                adversarial_input=(
                    adversarial_input.detach().cpu()
                    if adversarial_input is not None
                    else None
                ),
            )

    if not rows:
        raise RuntimeError(
            "No validation samples were processed."
        )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    summary = OrderedDict(
        num_samples=len(rows),
        attack_enabled=attack_enabled,
        clean_final_psnr=mean_or_nan(
            clean_final_psnr_values
        ),
        clean_final_ssim=mean_or_nan(
            clean_final_ssim_values
        ),
    )

    if attack_enabled:
        summary.update(
            attacked_final_psnr=mean_or_nan(
                attacked_final_psnr_values
            ),
            attacked_final_ssim=mean_or_nan(
                attacked_final_ssim_values
            ),
            average_psnr_drop=mean_or_nan(
                psnr_drop_values
            ),
        )

    summary_path = (
        result_root / "summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as summary_file:
        summary_file.write(
            "Predict-and-Refine Evaluation Summary\n"
        )
        summary_file.write(
            "=====================================\n"
        )

        summary_file.write(
            f"Number of samples: {summary['num_samples']}\n"
        )

        summary_file.write(
            f"Attack enabled: {summary['attack_enabled']}\n"
        )

        summary_file.write(
            f"Clean final PSNR: "
            f"{summary['clean_final_psnr']:.6f} dB\n"
        )

        summary_file.write(
            f"Clean final SSIM: "
            f"{summary['clean_final_ssim']:.8f}\n"
        )

        if attack_enabled:
            summary_file.write(
                f"Attacked final PSNR: "
                f"{summary['attacked_final_psnr']:.6f} dB\n"
            )

            summary_file.write(
                f"Attacked final SSIM: "
                f"{summary['attacked_final_ssim']:.8f}\n"
            )

            summary_file.write(
                f"Average PSNR drop: "
                f"{summary['average_psnr_drop']:.6f} dB\n"
            )

            summary_file.write(
                f"PGD target: "
                f"{attack_opt.get('target', 'prenet')}\n"
            )

            summary_file.write(
                f"PGD epsilon: "
                f"{float(attack_opt['epsilon_255']):.4f}/255\n"
            )

            summary_file.write(
                f"PGD step size: "
                f"{float(attack_opt['step_size_255']):.4f}/255\n"
            )

            summary_file.write(
                f"PGD steps: "
                f"{int(attack_opt['steps'])}\n"
            )

    logger.info(
        "Evaluation finished successfully."
    )

    logger.info(
        "Average clean final PSNR: %.4f dB",
        summary["clean_final_psnr"],
    )

    logger.info(
        "Average clean final SSIM: %.6f",
        summary["clean_final_ssim"],
    )

    if attack_enabled:
        logger.info(
            "Average attacked final PSNR: %.4f dB",
            summary["attacked_final_psnr"],
        )

        logger.info(
            "Average attacked final SSIM: %.6f",
            summary["attacked_final_ssim"],
        )

        logger.info(
            "Average PSNR drop: %.4f dB",
            summary["average_psnr_drop"],
        )

    logger.info(
        "Metrics CSV: %s",
        csv_path,
    )

    logger.info(
        "Summary: %s",
        summary_path,
    )


if __name__ == "__main__":
    main()