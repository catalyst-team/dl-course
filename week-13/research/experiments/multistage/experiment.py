import os
from pathlib import Path

from catalyst import utils
from catalyst.dl import ConfigExperiment
from dataset import CIFAR10, get_cat_dogs_dataset, get_reader
import torch


class Experiment(ConfigExperiment):
    def get_loaders(self, stage: str, **kwargs):
        loaders = dict()
        data_params = dict(self.stages_config[stage]["data_params"])
        data_path = Path(os.environ["DATA_PATH"])

        if stage == "stage1":
            for mode in ["train", "valid"]:
                dataset = CIFAR10(
                    root=(data_path / "data_cifar").as_posix(),
                    train=(mode == "train"),
                    download=True,
                    transform=self.get_transforms(stage=stage, dataset=mode),
                )
                loaders[mode] = utils.get_loader(
                    dataset,
                    open_fn=lambda x: x,
                    dict_transform=lambda x: x,
                    shuffle=(mode == "train"),
                    sampler=None,
                    drop_last=(mode == "train"),
                    **data_params,
                )
        elif stage == "stage2":
            data_path = (data_path / "data_cat_dogs").as_posix() + "/*"
            tag_file_path = (data_path / "cat_dog_labeling.json").as_posix()
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
