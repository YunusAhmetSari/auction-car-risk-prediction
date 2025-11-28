import os
import shutil

import kagglehub
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _move_files(source_path: str, destination: str, replace: bool = False) -> None:
    """
    Moves files from the source path to the destination directory.
    Args:
        source_path (str): The path where the files are currently located.
        destination (str): The directory where the files will be moved.
        replace (bool): If False, existing files won't be overwritten. If True, files will be replaced.
    Returns:
        None
    """
    data_dir = os.path.join(os.getcwd(), destination)
    os.makedirs(data_dir, exist_ok=True)

    files = os.listdir(source_path)
    for file_name in files:
        full_file_name = os.path.join(source_path, file_name)
        destination_file = os.path.join(data_dir, file_name)

        # if os.path.isfile(full_file_name):
        # Check if file already exists in destination
        if os.path.exists(destination_file) and not replace:
            print(f"File already exists, skipping: {file_name}")
            continue

        print(f"Moving file: {full_file_name} to {data_dir}")
        shutil.move(full_file_name, data_dir)

    print(f"Files moved to '{destination}' directory.")


def load_from_kaggle(
    dataset_link: str,
    destination: str = "",
    create_subfolder=True,
    replace: bool = False,
) -> list[str]:
    """
    Loads a dataset from Kaggle and moves it to the specified destination directory.
    Args:
        dataset_link (str): The Kaggle dataset link in the format 'username/dataset-name'

        destination (str): The directory where the dataset will be saved. Defaults to the dataset name.

        create_subfolder (bool): If True, creates a subfolder with the dataset name in the destination directory.

        replace (bool): If False, existing files/directories won't be overwritten. If True, they will be replaced.
    Returns:
        List [str]: A list of file names that were moved to the destination directory.
    """
    # Check if destination already exists and handle replace logic
    if create_subfolder:
        final_destination = os.path.join(destination, dataset_link.split("/")[-1])
    else:
        final_destination = destination

    full_destination_path = os.path.join(os.getcwd(), final_destination)

    # If destination exists and replace is False, return existing files without downloading
    if os.path.exists(full_destination_path) and not replace:
        existing_files = os.listdir(full_destination_path)
        if existing_files:  # If directory exists and contains files
            print(
                f"Destination directory '{final_destination}' already exists with files. Skipping download (replace=False)."
            )
            return existing_files

    # If replace is True and destination exists, remove it first
    if os.path.exists(full_destination_path) and replace:
        print(f"Removing existing destination directory: {final_destination}")
        shutil.rmtree(full_destination_path)

    # Download the dataset
    path = kagglehub.dataset_download(dataset_link, force_download=True)

    # Create destination directory
    if not os.path.exists(full_destination_path):
        os.makedirs(full_destination_path, exist_ok=True)

    print(f"Loading dataset from {path} to {final_destination}")

    _move_files(path, final_destination, replace)
    return os.listdir(full_destination_path)


def load_competition_from_kaggle(
    competition_name: str,
    destination: str = "",
    create_subfolder=True,
    replace: bool = False,
) -> list[str]:
    """
    Loads a competition dataset from Kaggle and moves it to the specified destination directory.
    Args:
        competition_name (str): The Kaggle competition name (e.g., 'DontGetKicked')

        destination (str): The directory where the dataset will be saved. Defaults to the competition name.

        create_subfolder (bool): If True, creates a subfolder with the competition name in the destination directory.

        replace (bool): If False, existing files/directories won't be overwritten. If True, they will be replaced.
    Returns:
        List [str]: A list of file names that were moved to the destination directory.
    """
    # Check if destination already exists and handle replace logic
    if create_subfolder:
        final_destination = os.path.join(destination, competition_name)
    else:
        final_destination = destination

    full_destination_path = os.path.join(os.getcwd(), final_destination)

    # If destination exists and replace is False, return existing files without downloading
    if os.path.exists(full_destination_path) and not replace:
        existing_files = os.listdir(full_destination_path)
        if existing_files:  # If directory exists and contains files
            print(
                f"Destination directory '{final_destination}' already exists with files. Skipping download (replace=False)."
            )
            return existing_files

    # If replace is True and destination exists, remove it first
    if os.path.exists(full_destination_path) and replace:
        print(f"Removing existing destination directory: {final_destination}")
        shutil.rmtree(full_destination_path)

    # Download the competition data
    path = kagglehub.competition_download(competition_name, force_download=True)

    # Create destination directory
    if not os.path.exists(full_destination_path):
        os.makedirs(full_destination_path, exist_ok=True)

    print(f"Loading competition data from {path} to {final_destination}")

    _move_files(path, final_destination, replace)
    return os.listdir(full_destination_path)


def clean_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Converts data types and cleans up categories in the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to be cleaned.
    Returns:
        df (pd.DataFrame): The cleaned DataFrame.
    """
    df = df.copy()

    # Konvertiert Datum von Unix-Timestamp zu Datetime-Objekt
    df["PurchDate"] = pd.to_datetime(df["PurchDate"])

    # Konvertiert fälschlicherweise numerische Features zu Objekt
    wrong_dtypes = ["WheelTypeID", "BYRNO", "VNZIP1", "IsOnlineSale"]
    for col in wrong_dtypes:
        df[col] = df[col].astype("str")

    # Vereinheitlicht und bereinigt Kategorien für mehr Konsistenz
    df["Transmission"] = df["Transmission"].replace("Manual", "MANUAL")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new features to the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame.
    Returns:
        df (pd.DataFrame): DataFrame with additional features.
    """
    df["CostPerMile"] = df["VehBCost"] / df["VehOdo"]
    df["WarrantyPerCost"] = df["WarrantyCost"] / df["VehBCost"]
    df["MilesPerYear"] = df["VehOdo"] / (df["VehicleAge"] + 1)

    return df


class TopNCategoriesTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer for bucketing high-cardinality categories."""

    def __init__(self, bucket_cols: list[str] | None = None, top_n: int = 20) -> None:
        self.bucket_cols = bucket_cols
        self.top_n = top_n
        self.bucket_dict_: dict[str, set[str]] = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X, copy=True)

        columns = (
            self.bucket_cols if self.bucket_cols is not None else df.columns.tolist()
        )

        self.bucket_dict_.clear()
        for col in columns:
            counts = df[col].value_counts()
            top_categories = counts.nlargest(self.top_n).index.tolist()
            self.bucket_dict_[col] = set(top_categories)

        return self

    def transform(self, X):
        df = pd.DataFrame(X, copy=True)

        for col in df.columns:
            top_categories = self.bucket_dict_.get(col, set())
            if top_categories:
                df[col] = df[col].where(df[col].isin(top_categories), "Other")

        return df

    def set_output(self, *, transform=None):
        """Compatibility for sklearn's set_output API."""
        return self


def memory_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimises the memory usage of the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to be optimised.
    Returns:
        df (pd.DataFrame): The optimised DataFrame.
    """
    df = df.copy()

    # Bestimmung der numerischen Features
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Optimiere Speicherverbrauch durch Datentyp Konvertierung auf "unsigned" sofern möglich
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], downcast="unsigned")

    return df


def drop_columns(
    df: pd.DataFrame,
    cols_to_drop: list,
) -> pd.DataFrame:
    """
    Feature Selection by dropping columns of the DataFrame.
    Args:
        df (pd.DataFrame): The original DataFrame.
        cols_to_drop (list): List of column names to remove.
    Returns:
        df (pd.DataFrame): The DataFrame with selected features.
    """
    df = df.copy()
    
    # errors='ignore' verhindert Abstürze, falls eine Spalte schon weg ist
    return df.drop(columns=cols_to_drop, errors="ignore")
