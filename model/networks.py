import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    if isinstance(
        m,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
        ),
    ):
        init.orthogonal_(
            m.weight.data,
            gain=1,
        )

        if m.bias is not None:
            init.constant_(
                m.bias.data,
                0.0,
            )

    elif isinstance(m, nn.Linear):
        init.orthogonal_(
            m.weight.data,
            gain=1,
        )

        if m.bias is not None:
            init.constant_(
                m.bias.data,
                0.0,
            )

    elif isinstance(
        m,
        (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
        ),
    ):
        if m.weight is not None:
            init.normal_(
                m.weight.data,
                mean=1.0,
                std=0.02,
            )

        if m.bias is not None:
            init.constant_(
                m.bias.data,
                0.0,
            )


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################
def _sampling_kwargs(model_opt, unet_opt):
    # Les nouveaux samplers concernent uniquement
    # le UNet 2D de SR3.
    if model_opt["which_model_G"] != "sr3":
        return {}

    fpdh_drop_prob = unet_opt.get(
        "fpdh_drop_prob",
        0.3,
    )

    if fpdh_drop_prob is None:
        fpdh_drop_prob = 0.3

    return {
        "downsample_type": unet_opt.get(
            "downsample_type",
            "original",
        ),
        "upsample_type": unet_opt.get(
            "upsample_type",
            "original",
        ),
        "fpdh_drop_prob": fpdh_drop_prob,
    }


def define_P(opt):
    model_opt = opt["model"]

    if model_opt["which_model_G"] == "ddpm":
        from .ddpm_modules import diffusion, unet

    elif model_opt["which_model_G"] == "sr3":
        from .sr3_modules import diffusion, unet

    else:
        raise ValueError(
            "Unknown which_model_G: "
            f"{model_opt['which_model_G']}"
        )

    prenet_opt = model_opt["unet"]["PreNet"]

    if (
        "norm_groups" not in prenet_opt
        or prenet_opt["norm_groups"] is None
    ):
        prenet_opt["norm_groups"] = 32

    sampling_kwargs = _sampling_kwargs(
        model_opt,
        prenet_opt,
    )

    model = unet.UNet(
        in_channel=prenet_opt["in_channel"],
        out_channel=prenet_opt["out_channel"],
        norm_groups=prenet_opt["norm_groups"],
        inner_channel=prenet_opt["inner_channel"],
        channel_mults=prenet_opt["channel_multiplier"],
        attn_res=prenet_opt["attn_res"],
        res_blocks=prenet_opt["res_blocks"],
        dropout=prenet_opt["dropout"],
        with_noise_level_emb=False,
        image_size=model_opt["diffusion"]["image_size"],
        **sampling_kwargs,
    )

    if opt["phase"] == "train":
        init_weights(
            model,
            init_type="orthogonal",
        )

    if opt["gpu_ids"] and opt["distributed"]:
        assert torch.cuda.is_available()
        model = nn.DataParallel(model)

    return model

# Generator
def define_G(opt):
    model_opt = opt["model"]

    if model_opt["which_model_G"] == "ddpm":
        from .ddpm_modules import diffusion, unet

    elif model_opt["which_model_G"] == "sr3":
        from .sr3_modules import diffusion, unet

    else:
        raise ValueError(
            "Unknown which_model_G: "
            f"{model_opt['which_model_G']}"
        )

    denoise_opt = model_opt["unet"]["DenoiseNet"]

    if (
        "norm_groups" not in denoise_opt
        or denoise_opt["norm_groups"] is None
    ):
        denoise_opt["norm_groups"] = 32

    sampling_kwargs = _sampling_kwargs(
        model_opt,
        denoise_opt,
    )

    model = unet.UNet(
        in_channel=denoise_opt["in_channel"],
        out_channel=denoise_opt["out_channel"],
        norm_groups=denoise_opt["norm_groups"],
        inner_channel=denoise_opt["inner_channel"],
        channel_mults=denoise_opt["channel_multiplier"],
        attn_res=denoise_opt["attn_res"],
        res_blocks=denoise_opt["res_blocks"],
        dropout=denoise_opt["dropout"],
        image_size=model_opt["diffusion"]["image_size"],
        **sampling_kwargs,
    )

    netG = diffusion.GaussianDiffusion(
        model,
        image_size=model_opt["diffusion"]["image_size"],
        channels=model_opt["diffusion"]["channels"],
        loss_type="l1",
        conditional=model_opt["diffusion"]["conditional"],
        schedule_opt=model_opt["beta_schedule"]["train"],
    )

    if opt["phase"] == "train":
        init_weights(
            netG,
            init_type="orthogonal",
        )

    if opt["gpu_ids"] and opt["distributed"]:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)

    return netG