import torch, os
import torch.nn as nn
import pandas as pd


from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from .paths import *
import logging
from .aug import get_default_aug


class UKBGlaucomaDataset(Dataset):
    def __init__(
        self,
        image_size: int,
        split: str,
        label_cols: List[str] = [
            "FundusImageGlaucomaLabel",
            # "LifeTimeGlaucomaLabel",
        ],
        label_weights: dict = {
            "FundusImageGlaucomaLabel": 1,
            # "LifeTimeGlaucomaLabel": 1,
        },
        df_path: str = GLAUCOMA_SPREADSHEET_PATH,
        image_dir: str = FUNDUS_DIR,
        transform: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.image_size = image_size
        self.label_cols = label_cols
        self.split = split
        self.label_weights = label_weights
        self.transform = (
            get_default_aug(image_size, split) if transform is None else transform
        )
        self.df = pd.read_csv(df_path)
        self.__df_preprocessing()

    def get_labels_for_balance(
        self,
    ):
        return torch.tensor(list(self.df["FundusImageGlaucomaLabel"])).long()

    def __df_preprocessing(self):
        # self.df = self.df[self.df[self.label_cols].notnull()]
        for label in self.label_cols:
            self.df[label] = self.df[label] == True
        self.num_classes = len(self.label_cols)
        self.df = self.df[self.df["split"] == self.split]

    def __len__(self):
        return len(self.df)

    def image_id_to_path(self, id):
        return os.path.join(self.image_dir, f"{id}{self.img_file_suffix}.png")

    def get_image(self, data):
        # return Image.open("./spreadsheets/test.png").convert("RGB")
        return Image.open(data["image_path"]).convert("RGB")

    def get_glaucoma_label(self, data):
        return torch.Tensor(data.loc[self.label_cols])

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.df.iloc[idx]  # 77202, 86903
        image, label = self.get_image(data), self.get_glaucoma_label(data)
        image = self.transform(image)  # (C, H, W)
        return image, label


class UKBGlaucomaTestDataset(Dataset):
    def __init__(
        self,
        image_size: int,
        split: str,
        label_cols: List[str] = ["DiagnosisBefore", "HasDiagnosis"],
        label_weights: dict = {
            "DiagnosisBefore": 1,
            "HasDiagnosis": 1,
        },
        df_path: str = GLAUCOMA_SPREADSHEET_PATH,
        image_dir: str = FUNDUS_DIR,
        transform: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.image_size = image_size
        self.label_cols = label_cols
        self.split = split
        self.label_weights = label_weights
        self.transform = (
            get_default_aug(image_size, split) if transform is None else transform
        )
        self.df = pd.read_csv("./spreadsheets/fundus.csv")
        self.__df_preprocessing()
        self.df = self.df.head(10000)

    def get_labels_for_balance(
        self,
    ):
        return torch.tensor(list(self.df[self.label_cols[0]])).long()

    def __df_preprocessing(self):
        # self.df = self.df[self.df[self.label_cols].notnull()]
        for label in self.label_cols:
            self.df[label] = self.df[label] == True
        self.num_classes = len(self.label_cols)
        self.df = self.df[self.df["split"] == self.split]

    def __len__(self):
        return len(self.df)

    def image_id_to_path(self, id):
        return os.path.join(self.image_dir, f"{id}{self.img_file_suffix}.png")

    def get_image(self, data):
        return Image.open("./spreadsheets/test.png").convert("RGB")

    def get_glaucoma_label(self, data):
        return torch.Tensor(data.loc[self.label_cols])

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.df.iloc[idx]
        image, label = self.get_image(data), self.get_glaucoma_label(data)
        image = self.transform(image)
        return image, label


def build_glaucoma_dataset(
    args,
    split: str,  # [train, val, test]
    **kwargs,
):
    if args.debug:
        logging.info("Debug dataset in used.")
        return UKBGlaucomaTestDataset(
            image_size=args.image_size,
            split=split,
            **kwargs,
        )

    logging.info("Official dataset in used.")
    return UKBGlaucomaDataset(
        image_size=args.image_size,
        split=split,
        **kwargs,
    )


def build_binary_glaucoma_datasets(args, **kwargs):
    train_dataset = build_glaucoma_dataset(
        args,
        split="train",
        label_cols=["FundusImageGlaucomaLabel"],
        label_weights={"FundusImageGlaucomaLabel": 1},
        **kwargs,
    )
    val_dataset = build_glaucoma_dataset(
        args,
        split="val",
        label_cols=["FundusImageGlaucomaLabel"],
        label_weights={"FundusImageGlaucomaLabel": 1},
        **kwargs,
    )
    test_dataset = build_glaucoma_dataset(
        args,
        split="test",
        label_cols=["FundusImageGlaucomaLabel"],
        label_weights={"FundusImageGlaucomaLabel": 1},
        **kwargs,
    )
    return train_dataset, val_dataset, test_dataset


def build_multitask_glaucoma_datasets(args, **kwargs):
    train_dataset = build_glaucoma_dataset(
        args,
        split="train",
        label_cols=[
            "FundusImageGlaucomaLabel",
            "LifeTimeGlaucomaLabel",
        ],
        label_weights={
            "FundusImageGlaucomaLabel": 1,
            "LifeTimeGlaucomaLabel": 1,
        },
        **kwargs,
    )
    val_dataset = build_glaucoma_dataset(
        args,
        split="val",
        label_cols=[
            "FundusImageGlaucomaLabel",
            "LifeTimeGlaucomaLabel",
        ],
        label_weights={
            "FundusImageGlaucomaLabel": 1,
            "LifeTimeGlaucomaLabel": 1,
        },
        **kwargs,
    )
    test_dataset = build_glaucoma_dataset(
        args,
        split="test",
        label_cols=[
            "FundusImageGlaucomaLabel",
            "LifeTimeGlaucomaLabel",
        ],
        label_weights={
            "FundusImageGlaucomaLabel": 1,
            "LifeTimeGlaucomaLabel": 1,
        },
        **kwargs,
    )
    return train_dataset, val_dataset, test_dataset
