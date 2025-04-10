#!/usr/bin/env python3
"""writer.py in src/base_repo/fileio/dataframe."""

from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger
from pyarrow import feather


def save_df_to_file(df: pd.DataFrame, output_file: Union[Path, str]) -> None:
    """
    Save a pandas DataFrame to a file in CSV, Parquet, or Feather format.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be saved.
    output_file : Union[Path, str]
        The file path where the DataFrame will be saved. The file extension must be
        one of `.csv`, `.parquet`, or `.feather` to determine the file format.

    Raises
    ------
    ValueError
        If the file extension is not one of the supported formats.

    Notes
    -----
    - For `.csv`, the DataFrame is saved with `index=False`.
    - For `.feather`, this uses `pyarrow.feather.write_feather`.
    - Logging is used to record the success or failure of the operation.
    """



    output_file = Path(output_file)
    
    if output_file.suffix == ".csv":
        df.to_csv(output_file, index=False)
    elif output_file.suffix == ".parquet":
        df.to_parquet(output_file)
    elif output_file.suffix == ".feather":
        feather.write_feather(df, output_file.as_posix())
    else:
        logger.error(f"Unsupported file format: {output_file.suffix}")
        raise ValueError(f"Unsupported file format: {output_file.suffix}")

    logger.info(f"Saved DataFrame to {output_file}")
