"""Decode a still image using the local Pillow installation."""

from PIL import Image, ImageOps, UnidentifiedImageError


def load_still(filename):
    try:
        with Image.open(filename) as original:
            if original.format in {"EPS", "WMF"}:
                raise ValueError("Choose a raster image such as PNG, JPEG or WebP.")
            return ImageOps.exif_transpose(original)
    except UnidentifiedImageError as error:
        raise ValueError("This file could not be opened as an image by the local runtime.") from error
