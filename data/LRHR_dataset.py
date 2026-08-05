from __future__ import annotations

import json
import pickle
import random
from io import BytesIO
from pathlib import Path

import lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


try:
    FLIP_LEFT_RIGHT = Image.Transpose.FLIP_LEFT_RIGHT
    FLIP_TOP_BOTTOM = Image.Transpose.FLIP_TOP_BOTTOM

    ROTATE_90 = Image.Transpose.ROTATE_90
    ROTATE_180 = Image.Transpose.ROTATE_180
    ROTATE_270 = Image.Transpose.ROTATE_270

except AttributeError:
    # Compatibilité avec les anciennes versions de Pillow.
    FLIP_LEFT_RIGHT = Image.FLIP_LEFT_RIGHT
    FLIP_TOP_BOTTOM = Image.FLIP_TOP_BOTTOM

    ROTATE_90 = Image.ROTATE_90
    ROTATE_180 = Image.ROTATE_180
    ROTATE_270 = Image.ROTATE_270


class LRHRDataset(Dataset):
    def __init__(
        self,
        dataroot,
        datatype,
        l_resolution=128,
        r_resolution=128,
        split="train",
        data_len=-1,
        need_LR=False,
    ):
        super().__init__()

        self.dataroot = Path(dataroot)
        self.datatype = str(datatype).lower()
        self.split = split
        self.need_LR = need_LR

        # Dans notre nouveau loader, r_resolution correspond
        # à la taille du crop d'entraînement.
        self.crop_size = int(r_resolution)

        if self.datatype != "lmdb":
            raise ValueError(
                "Le nouveau LRHRDataset attend datatype='lmdb', "
                f"reçu : {self.datatype}"
            )

        if not self.dataroot.is_dir():
            raise FileNotFoundError(
                f"LMDB introuvable : {self.dataroot}"
            )

        if self.crop_size <= 0:
            raise ValueError(
                "r_resolution doit être strictement positif."
            )

        # L'environnement utilisé dans __getitem__ est ouvert
        # paresseusement dans chaque worker du DataLoader.
        self.environment = None

        self.keys, self.metadata = self._read_index()

        self.dataset_len = len(self.keys)

        if data_len is None or data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(
                int(data_len),
                self.dataset_len,
            )

        if self.data_len <= 0:
            raise RuntimeError(
                f"Le LMDB ne contient aucun échantillon : {self.dataroot}"
            )

    def _open_environment(self):
        if self.environment is None:
            self.environment = lmdb.open(
                str(self.dataroot),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=True,
                max_readers=256,
            )

        return self.environment

    def _read_index(self):
        environment = lmdb.open(
            str(self.dataroot),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            subdir=True,
            max_readers=256,
        )

        try:
            with environment.begin(write=False) as transaction:
                raw_keys = transaction.get(b"__keys__")
                raw_length = transaction.get(b"__len__")
                raw_metadata = transaction.get(b"__meta__")

                if raw_keys is None:
                    raise RuntimeError(
                        f"Clé __keys__ absente dans {self.dataroot}"
                    )

                if raw_length is None:
                    raise RuntimeError(
                        f"Clé __len__ absente dans {self.dataroot}"
                    )

                keys = pickle.loads(raw_keys)
                declared_length = int(
                    raw_length.decode("utf-8")
                )

                if raw_metadata is None:
                    metadata = {}
                else:
                    metadata = json.loads(
                        raw_metadata.decode("utf-8")
                    )

                if declared_length != len(keys):
                    raise RuntimeError(
                        "Index LMDB incohérent : "
                        f"__len__={declared_length}, "
                        f"len(__keys__)={len(keys)}"
                    )

        finally:
            environment.close()

        return keys, metadata

    def __len__(self):
        return self.data_len

    @staticmethod
    def _decode_rgb(image_bytes):
        if image_bytes is None:
            raise RuntimeError(
                "Image absente dans le LMDB."
            )

        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            image = image.convert("RGB")

        return image

    def _read_pair(self, record_id):
        environment = self._open_environment()

        blur_key = (
            f"{record_id}/blur".encode("utf-8")
        )

        sharp_key = (
            f"{record_id}/sharp".encode("utf-8")
        )

        with environment.begin(write=False) as transaction:
            blur_bytes = transaction.get(blur_key)
            sharp_bytes = transaction.get(sharp_key)

        blur = self._decode_rgb(blur_bytes)
        sharp = self._decode_rgb(sharp_bytes)

        if blur.size != sharp.size:
            raise RuntimeError(
                "Dimensions différentes dans la paire "
                f"{record_id} : blur={blur.size}, sharp={sharp.size}"
            )

        return blur, sharp

    def _random_aligned_crop(
        self,
        blur,
        sharp,
    ):
        width, height = blur.size

        if width < self.crop_size:
            raise ValueError(
                f"Largeur insuffisante : {width} "
                f"< crop_size={self.crop_size}"
            )

        if height < self.crop_size:
            raise ValueError(
                f"Hauteur insuffisante : {height} "
                f"< crop_size={self.crop_size}"
            )

        left = random.randint(
            0,
            width - self.crop_size,
        )

        top = random.randint(
            0,
            height - self.crop_size,
        )

        crop_box = (
            left,
            top,
            left + self.crop_size,
            top + self.crop_size,
        )

        return (
            blur.crop(crop_box),
            sharp.crop(crop_box),
        )

    @staticmethod
    def _aligned_augmentation(
        blur,
        sharp,
    ):
        # Flip horizontal.
        if random.random() < 0.5:
            blur = blur.transpose(
                FLIP_LEFT_RIGHT
            )
            sharp = sharp.transpose(
                FLIP_LEFT_RIGHT
            )

        # Flip vertical.
        if random.random() < 0.5:
            blur = blur.transpose(
                FLIP_TOP_BOTTOM
            )
            sharp = sharp.transpose(
                FLIP_TOP_BOTTOM
            )

        # Rotation commune parmi 0°, 90°, 180° et 270°.
        rotation = random.randint(0, 3)

        if rotation == 1:
            blur = blur.transpose(ROTATE_90)
            sharp = sharp.transpose(ROTATE_90)

        elif rotation == 2:
            blur = blur.transpose(ROTATE_180)
            sharp = sharp.transpose(ROTATE_180)

        elif rotation == 3:
            blur = blur.transpose(ROTATE_270)
            sharp = sharp.transpose(ROTATE_270)

        return blur, sharp

    def __getitem__(self, index):
        record_id = self.keys[index]

        blur, sharp = self._read_pair(
            record_id
        )

        if self.split == "train":
            blur, sharp = self._random_aligned_crop(
                blur,
                sharp,
            )

            blur, sharp = self._aligned_augmentation(
                blur,
                sharp,
            )

        # TF.to_tensor convertit RGB HWC uint8 en
        # float32 CHW dans l'intervalle [0, 1].
        #
        # Le papier ne précise pas le choix [0,1] ou [-1,1].
        # [0,1] est cohérent avec les utilitaires existants
        # de ce dépôt.
        img_blur = TF.to_tensor(blur)
        img_sharp = TF.to_tensor(sharp)

        result = {
            "SR": img_blur,
            "HR": img_sharp,
            "Index": index,
        }

        if self.need_LR:
            result["LR"] = img_blur

        return result

    def __getstate__(self):
        # Empêche le partage d'un environnement LMDB ouvert
        # lors de la création des workers.
        state = self.__dict__.copy()
        state["environment"] = None
        return state

    def __del__(self):
        environment = getattr(
            self,
            "environment",
            None,
        )

        if environment is not None:
            environment.close()
            self.environment = None