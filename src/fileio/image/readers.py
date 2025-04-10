#!/usr/bin/env python3
"""Wrapper functions for loading images using OpenCV, PIL, and other libraries.

This module provides functionality to load images using OpenCV in a Pythonic
way. It ensures that images are loaded efficiently and supports different color
spaces. It also includes error handling to manage exceptions gracefully.
"""
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import PIL.ImageFile
from loguru import logger
from PIL.PngImagePlugin import PngInfo  # noqa: F401

from base_repo.fileio.text import is_empty_file
from base_repo.processing.image.color_icc import build_apply_icc


# PIL settings
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = False
PIL.PngImagePlugin.MAX_TEXT_CHUNK   = 2048 * 2048


cv2.setUseOptimized(True)


def cv2_loader(
    filepath: Union[str, Path],
    flag: int = cv2.IMREAD_UNCHANGED,
    apply_icc: bool = False,
) -> np.array:
    """Load an image from the specified path using OpenCV.

    Args:
        path (str): The path to the image file.
        flag (int): The flag specifying the color space and channel format of the
        image. Defaults to cv2.IMREAD_UNCHANGED.

    Returns:
        np.ndarray: The loaded image as a NumPy array.

    Raises:
        FileNotFoundError: If the image file does not exist.
        Exception: If there is an error during the loading process.
    """
    if is_empty_file(filepath):
        raise FileNotFoundError(f"Image file is empty: {filepath}")

    try:
        img = cv2.imread(str(filepath), flag)
        if img is None:
            raise FileNotFoundError(f"Image file not found at {filepath}.")

        if flag == cv2.IMREAD_COLOR or (len(img.shape) == 3 and img.shape[2] == 3):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if apply_icc:
            # read image with PIL
            img = pil_loader(filepath, apply_icc=apply_icc)

        return img
    except Exception as e:
        logger.error(f"Failed to load image from {filepath}: {e}")
        raise e


def pil_loader(filepath: Union[str, Path], apply_icc: bool = False) -> np.array:
    """Load an image from the specified path using PIL.

    Args:
        path (str): The path to the image file.

    Returns:
        np.ndarray: The loaded image as a NumPy array.

    Raises:
        FileNotFoundError: If the image file does not exist.
        Exception: If there is an error during the loading process.
    """
    try:
        img = PIL.Image.open(filepath)

        if apply_icc and img.info.get("icc_profile"):
            # read and apply ICC profile to image
            icc_bytes = img.info.get("icc_profile")
            img = build_apply_icc(icc_bytes, img=np.array(img))
        elif apply_icc and img.info.get("icc_profile") is None:
            logger.warning(f"No ICC profile found for image: {filepath}")

        return np.array(img)
    except FileNotFoundError as e:
        logger.error(f"Image file not found at {filepath}.")
        raise e
    except Exception as e:
        logger.error(f"Failed to load image from {filepath}: {e}")
        raise e
