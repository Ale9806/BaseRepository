#!/usr/bin/env python3
"""writers.py in src/base_repo/fileio/image."""
# flake8: noqa: B950
import re
from pathlib import Path
from typing import Optional
from typing import Union

import cv2
from loguru import logger
from PIL.PngImagePlugin import PngInfo

from base_repo.processing import timestamp


cv2.setUseOptimized(True)

re_fields = re.compile(r"<(.*?)>")
re_groups = re.compile(r"\((.*?)\)")

# regex patterns for filename parsing
REGEX_COMPILED = {
    "base_repo": {
        "fileparts": r"(?P<index>\d+)_(?P<uuid>[a-f0-9]{8})_(?P<patient_id>[a-zA-Z0-9\-]+)_split-(?P<split>[a-z]+)_(?P<label_name>[a-z0-9\-]+)\.(?P<ext>png|jpg|jpeg|gif|bmp|tiff|tif)",
        "misc_info": r"",
    },
    "filename_1": {
        "fileparts": r"(?P<prefix>(?P<id>[a-f0-9]{16})_part-(?P<block>[A-Z]\d{1,2}(-\d)?)_((?P<stain>HE|IPOX)-(?P<antibody>[A-Za-z0-9]+))_(?P<scan_id>\d{5,6}))(?P<suffix>_(?P<registration>fixed|moving)_x(?P<x_coord>\d+)_(x|y)(?P<y_coord>\d+))?\.(png|jpg|svs)",  # noqa E501
        "misc_info": r"",
    },
    "filename_2": {
        "fileparts": r"^(?P<prefix>(?P<id>[A-Z]{2,3}-\d{2}-\d{5,6})_(part-(?P<part>[A-Z]\d*))?)(_(?P<misc_info>[a-zA-Z0-9-]+(?:_[a-zA-Z0-9-]+)*))?_(?P<suffix>(?P<magnification>\d+x)_(?P<image_num>\d{3})?\.(?P<ext>jpg|png|svs)$)",  # noqa E501
        "misc_info": r"",
    },
    "filename_3": {
        "fileparts": r"^(?P<prefix>(?P<id>[a-f0-9]{16})_(part-(?P<part>[A-Z]\d*))?)(_(?P<misc_info>[a-zA-Z0-9-]+(?:_[a-zA-Z0-9-]+)*))?_(?P<suffix>(?P<magnification>\d+x)_(?P<image_num>\d{3})?\.(?P<ext>jpg|png|svs)$)",  # noqa E501
        "misc_info": r"(?P<misc_info>[a-zA-Z0-9-]+(?:_[a-zA-Z0-9-]+)*)",
    },
    # "misc_info": r"(?P<stain>HE|EVG|ki67)|(?P<mitosis>mitosis|mitoses)|(?P<who_grade>who-\d|who\d(?:-\d)?)",
    # "pending": r"pending|PEND",
}

# compile nested dictionary of regex patterns
REGEX_COMPILED = {
    key: {k: re.compile(v) for k, v in value.items()}
    for key, value in REGEX_COMPILED.items()
}


def create_png_metadata(
    image_file: Union[str, Path],
    data: dict,
    filename_regex: Optional[str] = "base_repo",
    ignore_case: Optional[bool] = False,
) -> PngInfo:
    """Create PNG metadata with the given data and optional filename parsing.

    Args:
        image_file (Union[str, Path]): The path to the image file.
        data (dict): The data to be added to the PNG metadata.
        filename_regex (Optional[str], optional): The regular expression pattern to parse the filename. Defaults to None.
        ignore_case (Optional[bool], optional): Whether to ignore case when matching the filename regex. Defaults to False.
        compute_checksum (Optional[str], optional): Whether to compute the checksum. Defaults to "md5".

    Returns:
        PngInfo: The PNG metadata.

    Examples:
        >>> image_file = "image.png"
        >>> data = {"key": "value"}
        >>> create_png_metadata(image_file,data)
        <PngInfo object at 0x...>
    """
    png_metadata = PngInfo()

    # add data to png metadata
    for key, value in data.items():
        if value is None:
            continue
        elif isinstance(value, str):
            if (
                not value
                or value.isspace()
                or value.lower() in {"na", "nan", "none", "null"}
            ):
                continue
        if isinstance(value, list):
            value = str(value[0]) if len(value) == 1 else ", ".join(map(str, value))
        elif isinstance(value, (int, float, dict)):
            value = str(value)

        png_metadata.add_text(key, str(value))

    # parse filename using regex and add to png metadata
    if filename_regex in REGEX_COMPILED:
        filename_regex = REGEX_COMPILED[filename_regex]

    if filename_regex and isinstance(filename_regex, (re.Pattern, str)):
        # if regex has fields <> then use them to parse filename
        if re_fields.search(filename_regex):
            filename_parts = re.findall(
                filename_regex, image_file, re.IGNORECASE if ignore_case else 0
            )
            for name, part in enumerate(filename_parts):
                png_metadata.add_text(f"{name}", part)

        elif re_groups.search(filename_regex):
            filename_parts = re.findall(
                filename_regex, image_file, re.IGNORECASE if ignore_case else 0
            )
            for name, part in enumerate(filename_parts):
                if isinstance(part, tuple):
                    logger.warning(f"Part: {part}")

                png_metadata.add_text(f"{name}", part)

    # add updated time to png metadata
    png_metadata.add_text("Last_updated", str(timestamp()))

    return png_metadata


def cv_writer(
    image_file: Union[str, Path],
    image: Union[str, Path],
    data: dict,
    filename_regex: Optional[str] = None,
    ignore_case: Optional[bool] = False,
    is_bgr: Optional[bool] = False,
) -> None:
    """Write image to file with given data and optional filename parsing.

    Args:
        image_file (Union[str, Path]): The path to the image file.
        image (Union[str, Path]): The image to be written to the file.
        data (dict): The data to be added to the PNG metadata.
        filename_regex (Optional[str], optional): The regular expression pattern to parse the filename. Defaults to None.
        ignore_case (Optional[bool], optional): Whether to ignore case when matching the filename regex. Defaults to False.

    Returns:
        None

    Examples:
        >>> image_file = "image.png"
        >>> image = "image.png"
        >>> data = {"key": "value"}
        >>> cv_writer(image_file, image, data)
    """
    # create png metadata
    png_metadata = create_png_metadata(image_file, data, filename_regex, ignore_case)

    # check for alpha channel and convert to BGR
    if image.shape[2] == 4:
        conversion_flag = cv2.COLOR_BGRA2BGR if is_bgr else cv2.COLOR_BGRA2RGBA
        image = cv2.cvtColor(image, conversion_flag)
    elif image.shape[2] == 3:
        image = image if is_bgr else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # check image is CV_8U or CV_16U
    if image.dtype not in [cv2.CV_8U, cv2.CV_16U]:
        raise ValueError(f"Invalid image type: {image.dtype}")

    # convert to BGR if not already
    if not is_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # write image to file with png metadata
    cv2.imwrite(
        image_file,
        image,
        [int(cv2.IMWRITE_PNG_COMPRESSION), 9],
        png_metadata=png_metadata,
    )
