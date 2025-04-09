#!/usr/bin/env python3
"""readers.py in src/base_repo/fileio/text."""

import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Union


import ujson as json
import yaml
from loguru import logger

from base_repo.fileio.text import is_empty_file
from base_repo.fileio.text import valid_file_ext


def json_loader(
    filepath: Union[str, Path, os.PathLike], strict: bool = False
) -> list[Any]:
    """Load JSON file and return its contents as a dictionary.

    Args:
        filepath (Union[str, Path]): The path to the JSON file.

    Returns:
        Dict: The contents of the JSON file as a dictionary.

    Examples:
        >>> json_loader("data.json")
        {'key': 'value'}
    """
    filepath = Path(filepath)
    if not valid_file_ext(filepath, [".json", ".jsonl"]):
        raise ValueError(f"Invalid file type: {filepath.suffix}")

    if is_empty_file(filepath, strict=strict):
        logger.error(f"File is empty: {filepath}")
        raise ValueError(f"File is empty: {filepath}")

    # load json and return as dict
    if filepath.suffix == ".json":
        try:
            with open(filepath) as f:
                data_dict = json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file: {filepath}")
            raise e
    elif filepath.suffix == ".jsonl":
        with open(filepath) as f:
            data_dict = [json.loads(line) for line in f if line.strip()]
    else:
        raise ValueError(f"Invalid file type: {filepath.suffix}")

    if not data_dict:
        raise ValueError(f"File is empty: {filepath}")
    else:
        return data_dict


def yaml_loader(filepath: Union[str, Path, os.PathLike]) -> Dict:
    """Load YAML file and return its contents as a dictionary.

    Args:
        filepath (Union[str, Path]): The path to the YAML file.

    Returns:
        Dict: The contents of the YAML file as a dictionary.

    Raises:
        ValueError: If file_path is None or empty.
        ValueError: If file_path has an invalid file type.
        FileNotFoundError: If file_path does not exist.

    Examples:
        >>> yaml_loader("config.yaml")
        {'key': 'value'}
    """
    filepath = Path(filepath)
    if not valid_file_ext(filepath, {".yaml", ".yml"}):
        logger.error(f"Invalid file type: {filepath.suffix}")
        raise ValueError(f"Invalid file type: {filepath.suffix}")

    if is_empty_file(filepath):
        logger.error(f"File is empty: {filepath}")
        raise ValueError(f"File is empty: {filepath}")

    # load yaml and return as dict
    with open(filepath) as file:
        data_dict = yaml.safe_load(file)

    if data_dict:
        return data_dict

    logger.error(f"File is empty: {filepath}")
    raise ValueError(f"File is empty: {filepath}")
