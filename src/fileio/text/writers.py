#!/usr/bin/env python3
"""writers.py in src/biovlmdata/fileio/text."""

import os
import pprint
from pathlib import Path
from pprint import pprint
from typing import Dict
from typing import Union

import pandas as pd
import ujson as json
import yaml
from loguru import logger
from pyarrow import feather

from base_repo.fileio.text import is_none_or_empty
from base_repo.fileio.text import make_dir
from base_repo.fileio.text import valid_file_ext


def yaml_writer(data: dict, file_path: Union[str, Path]) -> None:
    """Write data to a YAML file.

    Args:
        data (Dict): The data to be written to the YAML file.
        file_path (Union[str, Path]): The path to the YAML file.

    Returns:
        None

    Raises:
        ValueError: If data is None or empty.
        ValueError: If file_path is None or empty.

    Examples:
        >>> data = {'key': 'value'}
        >>> yaml_writer(data, "output.yaml")
    """
    if is_none_or_empty(data):
        raise ValueError("data cannot be None or empty")

    if is_none_or_empty(file_path):
        raise ValueError("file_path cannot be None or empty")

    # create directory if it does not exist
    file_path = make_dir(file_path)

    # write data to yaml file
    with open(file_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def is_json_serializable(data: dict) -> bool:
    """Check if the provided dictionary is serializable to JSON.

    Args:
        data: A dictionary containing the data to be checked for JSON serialization.

    Returns:
        bool: True if the data is JSON serializable, False otherwise.
    """
    serializable = True
    try:
        json.dumps(data)
    except Exception as e:
        # print error type without <class '...'>
        error_type = str(type(e)).split("'")[1]
        logger.error(f"{error_type}: Error serializing json data: {str(e)}")
        pprint(data)
        serializable = False

    return serializable


def json_writer(data: dict, filepath: Union[str, Path], **kwargs) -> None:
    """Write data to a JSON file.

    Args:
        data (Dict): The data to be written to the JSON file.
        filepath (Union[str, Path]): The path to the JSON file.

    Returns:
        None

    Raises:
        ValueError: If data is None or empty.
        ValueError: If file_path is None or empty.

    Examples:
        >>> data = {'key': 'value'}
        >>> json_writer(data, "output.json")
    """
    if is_none_or_empty(data):
        raise ValueError("data cannot be None or empty")

    if is_none_or_empty(filepath):
        raise ValueError("file_path cannot be None or empty")

    if not is_json_serializable(data):
        raise ValueError(f"Data for file {filepath.name} is not JSON serializable")

    # create directory if it does not exist
    filepath = make_dir(filepath)

    # write data to json file
    with open(filepath, "w") as f:
        if kwargs.get("indent", False):
            json.dump(data, f, indent=kwargs["indent"])
        else:
            json.dump(data, f)

    if filepath.stat().st_size == 0:
        raise ValueError(f"Error writing JSON file: {filepath} (size: 0)")



def jsonl_writer(data: list, path: str) -> None:
    """Save a list of dictionaries into a JSONL (JSON Lines) file.

    Parameters
    ----------
    data : list
        List of dictionaries to be saved into the JSONL file.
    path : Union[os.PathLike, Path]
        Path to save the JSONL file.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `data` is not a list.
    """
    if not isinstance(data, list):
        raise TypeError("`data` must be a list of dictionaries.")

    with open(path, "w") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")


def df_writer(df: pd.DataFrame, filepath: Union[str, Path, os.PathLike]) -> bool:
    """Write a pandas DataFrame to a file."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.error("DataFrame is empty")
        return False

    filepath = Path(filepath)
    if not valid_file_ext(filepath, {".csv", ".parquet", ".feather"}):
        logger.error(f"Invalid file type: {filepath.suffix}")
        raise ValueError(f"Invalid file type: {filepath.suffix}")

    if filepath.suffix == ".csv":
        df.to_csv(filepath, index=False)
    elif filepath.suffix == ".parquet":
        df.to_parquet(filepath, index=False)
    elif filepath.suffix == ".feather":
        feather.write_feather(df, filepath.as_posix())
    else:
        logger.error(f"Unsupported file format: {filepath.suffix}")
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
