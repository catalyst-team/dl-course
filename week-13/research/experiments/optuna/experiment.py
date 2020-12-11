import typing as tp
import os
from pathlib import Path

from catalyst import utils
from catalyst.dl import ConfigExperiment
from dataset import get_cat_dogs_dataset, get_reader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Experiment(ConfigExperiment):
    def get_loaders(self, stage: str, **kwargs) -> tp.Dict[str, DataLoader]:
        loaders = dict()
        data_params = dict(self.stages_config[stage]["data_params"])
        data_path = (
            Path(os.getenv("DATA_PATH")) / "data_cat_dogs"
        ).as_posix() + "/*"
        tag_file_path = (
            Path(os.getenv("DATA_PATH")) / "cat_dog_labeling.json"
        ).as_posix()
        train_data, valid_data, num_classes = get_cat_dogs_dataset(
            data_path, tag_file_path=tag_file_path
        )

        open_fn = get_reader(num_classes)
        data = [("train", train_data), ("valid", valid_data)]
        for mode, part in data:
            data_transform = self.get_transforms(stage=stage, dataset=mode)
            loaders[mode] = utils.get_loader(
                part,
                open_fn=open_fn,
                dict_transform=data_transform,
                shuffle=(mode == "train"),
                sampler=None,
                drop_last=(mode == "train"),
                **data_params,
            )

        return loaders
