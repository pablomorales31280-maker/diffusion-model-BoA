#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import lmdb
from PIL import Image
from tqdm import tqdm


GIB = 1024**3
MIB = 1024**2


def discover_pairs(
    dataset_root: Path,
    split: str,
) -> list[tuple[Path, Path]]:
    split_root = dataset_root / split

    if not split_root.is_dir():
        raise FileNotFoundError(
            f"Split GoPro introuvable : {split_root}"
        )

    blur_paths = sorted(
        split_root.glob("*/blur/*.png")
    )

    if not blur_paths:
        raise RuntimeError(
            f"Aucune image blur trouvée dans {split_root}"
        )

    pairs: list[tuple[Path, Path]] = []
    missing_sharp: list[tuple[Path, Path]] = []

    for blur_path in blur_paths:
        sharp_path = (
            blur_path.parent.parent
            / "sharp"
            / blur_path.name
        )

        if not sharp_path.is_file():
            missing_sharp.append(
                (blur_path, sharp_path)
            )
            continue

        pairs.append(
            (blur_path, sharp_path)
        )

    if missing_sharp:
        message = [
            f"{len(missing_sharp)} images sharp manquantes."
        ]

        for blur_path, sharp_path in missing_sharp[:10]:
            message.append(
                f"Blur : {blur_path}\n"
                f"Sharp attendu : {sharp_path}"
            )

        raise RuntimeError("\n".join(message))

    return pairs


def inspect_pair(
    blur_path: Path,
    sharp_path: Path,
    crop_size: int,
) -> tuple[int, int]:
    with Image.open(blur_path) as blur_image:
        blur_mode = blur_image.mode
        blur_size = blur_image.size

    with Image.open(sharp_path) as sharp_image:
        sharp_mode = sharp_image.mode
        sharp_size = sharp_image.size

    if blur_size != sharp_size:
        raise ValueError(
            "Dimensions différentes pour la paire :\n"
            f"Blur  : {blur_path} -> {blur_size}\n"
            f"Sharp : {sharp_path} -> {sharp_size}"
        )

    if blur_mode != "RGB":
        raise ValueError(
            f"Image blur non RGB : {blur_path}, "
            f"mode={blur_mode}"
        )

    if sharp_mode != "RGB":
        raise ValueError(
            f"Image sharp non RGB : {sharp_path}, "
            f"mode={sharp_mode}"
        )

    width, height = blur_size

    if width < crop_size or height < crop_size:
        raise ValueError(
            "Image trop petite pour le crop demandé :\n"
            f"Image : {blur_path}\n"
            f"Taille : {width}x{height}\n"
            f"Crop : {crop_size}x{crop_size}"
        )

    return width, height


def estimate_map_size(
    pairs: list[tuple[Path, Path]],
    requested_map_size_gb: float | None,
) -> int:
    if requested_map_size_gb is not None:
        if requested_map_size_gb <= 0:
            raise ValueError(
                "--map-size-gb doit être strictement positif."
            )

        return int(
            requested_map_size_gb * GIB
        )

    source_bytes = sum(
        blur_path.stat().st_size
        + sharp_path.stat().st_size
        for blur_path, sharp_path in pairs
    )

    # Les PNG sont conservés sous forme compressée.
    # Le facteur 1.5 et les 512 MiB supplémentaires
    # laissent de la place aux clés et métadonnées LMDB.
    estimated_size = int(
        source_bytes * 1.5 + 512 * MIB
    )

    return max(
        estimated_size,
        GIB,
    )


def put_bytes(
    transaction: lmdb.Transaction,
    key: str,
    value: bytes,
) -> None:
    success = transaction.put(
        key.encode("utf-8"),
        value,
        overwrite=False,
    )

    if not success:
        raise RuntimeError(
            f"Impossible d'écrire la clé LMDB : {key}"
        )


def build_lmdb(
    dataset_root: Path,
    split: str,
    output_path: Path,
    crop_size: int,
    commit_every: int,
    overwrite: bool,
    map_size_gb: float | None,
) -> None:
    pairs = discover_pairs(
        dataset_root=dataset_root,
        split=split,
    )

    print(f"Split                 : {split}")
    print(f"Nombre de paires      : {len(pairs)}")
    print(f"Racine GoPro          : {dataset_root}")
    print(f"Destination LMDB      : {output_path}")
    print(f"Crop minimal vérifié  : {crop_size}x{crop_size}")

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} existe déjà. "
                "Utilise --overwrite pour le remplacer."
            )

        print(
            f"Suppression de l'ancien LMDB : {output_path}"
        )

        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    map_size = estimate_map_size(
        pairs=pairs,
        requested_map_size_gb=map_size_gb,
    )

    print(
        "Map size LMDB         : "
        f"{map_size / GIB:.2f} GiB"
    )

    environment = lmdb.open(
        str(output_path),
        map_size=map_size,
        subdir=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        map_async=True,
        max_readers=256,
    )

    record_ids: list[str] = []
    transaction = environment.begin(write=True)

    try:
        for index, (blur_path, sharp_path) in enumerate(
            tqdm(
                pairs,
                desc=f"Création {split}.lmdb",
            )
        ):
            width, height = inspect_pair(
                blur_path=blur_path,
                sharp_path=sharp_path,
                crop_size=crop_size,
            )

            record_id = f"{index:08d}"
            record_ids.append(record_id)

            sequence_name = (
                blur_path.parent.parent.name
            )

            frame_name = blur_path.stem

            blur_relative_path = str(
                blur_path.relative_to(dataset_root)
            )

            sharp_relative_path = str(
                sharp_path.relative_to(dataset_root)
            )

            sample_metadata = {
                "record_id": record_id,
                "split": split,
                "sequence": sequence_name,
                "frame": frame_name,
                "blur_path": blur_relative_path,
                "sharp_path": sharp_relative_path,
                "width": width,
                "height": height,
                "channels": 3,
                "encoding": "png",
            }

            put_bytes(
                transaction,
                f"{record_id}/blur",
                blur_path.read_bytes(),
            )

            put_bytes(
                transaction,
                f"{record_id}/sharp",
                sharp_path.read_bytes(),
            )

            put_bytes(
                transaction,
                f"{record_id}/meta",
                json.dumps(
                    sample_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )

            if (
                (index + 1) % commit_every == 0
            ):
                transaction.commit()
                transaction = environment.begin(
                    write=True
                )

        dataset_metadata = {
            "format_version": 1,
            "dataset": "GoPro",
            "split": split,
            "num_samples": len(record_ids),
            "storage": "paired_png_bytes",
            "channels": 3,
            "crop_size": crop_size,
            "records": {
                "blur": "<record_id>/blur",
                "sharp": "<record_id>/sharp",
                "meta": "<record_id>/meta",
            },
        }

        put_bytes(
            transaction,
            "__keys__",
            pickle.dumps(
                record_ids,
                protocol=pickle.HIGHEST_PROTOCOL,
            ),
        )

        put_bytes(
            transaction,
            "__len__",
            str(len(record_ids)).encode("utf-8"),
        )

        put_bytes(
            transaction,
            "__meta__",
            json.dumps(
                dataset_metadata,
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
        )

        transaction.commit()

    except Exception:
        transaction.abort()
        environment.close()
        raise

    environment.sync()

    statistics = environment.stat()

    environment.close()

    print()
    print("Création terminée.")
    print(f"Entrées LMDB : {statistics['entries']}")
    print(f"Échantillons : {len(record_ids)}")
    print(f"Destination  : {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crée un LMDB de paires blur/sharp "
            "à partir du dataset GoPro classique."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Dossier contenant les sous-dossiers "
            "train et test."
        ),
    )

    parser.add_argument(
        "--split",
        choices=("train", "test"),
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--commit-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--map-size-gb",
        type=float,
        default=None,
        help=(
            "Taille maximale LMDB en GiB. "
            "Par défaut, elle est estimée automatiquement."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    arguments = parser.parse_args()

    if arguments.crop_size <= 0:
        parser.error(
            "--crop-size doit être strictement positif."
        )

    if arguments.commit_every <= 0:
        parser.error(
            "--commit-every doit être strictement positif."
        )

    return arguments


if __name__ == "__main__":
    args = parse_arguments()

    build_lmdb(
        dataset_root=args.dataset_root.resolve(),
        split=args.split,
        output_path=args.output.resolve(),
        crop_size=args.crop_size,
        commit_every=args.commit_every,
        overwrite=args.overwrite,
        map_size_gb=args.map_size_gb,
    )