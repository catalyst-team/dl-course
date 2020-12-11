from typing import Tuple
from enum import Enum

from common import app
from fastapi import File, UploadFile
import imageio
from pydantic import BaseModel
import services


class Orientation(Enum):
    portrait = "portrait"
    landscape = "landscape"


class ImageResult(BaseModel):
    label: str
    square: int
    orientation: Orientation
    shape: Tuple[int, int]


class ClassifyModelResult(BaseModel):
    tag: str


@app.post("/image_info", response_model=ImageResult)
async def get_image_info(
    label: str, image: UploadFile = File(...),  # noqa: B008
):
    img = imageio.imread(await image.read())
    height, width = (await services.get_shape.call(img))[:2]
    o = Orientation.landscape if width > height else Orientation.portrait
    return ImageResult(
        label=label,
        square=width * height,
        orientation=o,
        shape=(width, height),
    )


@app.post("/classify", response_model=ClassifyModelResult)
async def classify(image: UploadFile = File(...)):  # noqa: B008
    img = imageio.imread(await image.read())
    return ClassifyModelResult(
        tag=await services.ClassifyModel().predict.call(img),
    )
