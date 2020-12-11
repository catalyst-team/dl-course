import typing as tp
import json

from catalyst import utils
from catalyst.data import ReaderCompose, ScalarReader
from catalyst.data.cv.reader import ImageReader
import numpy as np
import torchvision


def get_cat_dogs_dataset(
    dirs: str = "/app/data/data_cat_dogs/*",
    extension: str = "*.jpg",
    test_size: float = 0.2,
    random_state: int = 42,
    tag_file_path: tp.Optional[str] = None,
) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, tp.Any], int]:
    dataset = utils.create_dataset(dirs=dirs, extension=extension)
    df = utils.create_dataframe(dataset, columns=["class", "filepath"])

    tag_to_label = utils.get_dataset_labeling(df, "class")
    if tag_file_path is not None:
        with open(tag_file_path, "w") as file:
            json.dump(tag_to_label, file)

    df_with_labels = utils.map_dataframe(
        df,
        tag_column="class",
        class_column="label",
        tag2class=tag_to_label,
        verbose=False,
    )

    train_data, valid_data = utils.split_dataframe_train_test(
        df_with_labels, test_size=test_size, random_state=random_state
    )
    return (
        train_data.to_dict("records"),
        valid_data.to_dict("records"),
        len(tag_to_label),
    )


def get_reader(num_classes: int = 2) -> ReaderCompose:
    return ReaderCompose(
        [
            ImageReader(
                input_key="filepath", output_key="image", rootpath="."
            ),
            ScalarReader(
                input_key="label",
                output_key="targets",
                default_value=-1,
                dtype=np.int64,
            ),
            ScalarReader(
                input_key="label",
                output_key="targets_one_hot",
                default_value=-1,
                dtype=np.int64,
                one_hot_classes=num_classes,
            ),
        ]
    )
