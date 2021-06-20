"""Utilities for working with OpenML."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
import os
from typing import List

import pandas as pd

from automl import dataset


@dataclass
class OpenMLTask:
    """A container used to store OpenML tasks."""

    df: pd.DataFrame
    feature_cols: List[str]
    numerical_cols: List[str]
    categorical_cols: List[str]
    label_col: str
    fold_col: str

    def dump(self, path: str) -> None:
        with open(path, "wb") as file_:
            pickle.dump(self, file_)

    @staticmethod
    def load(path: str) -> OpenMLTask:
        with open(path, "rb") as file_:
            return pickle.load(file_)


def task_path(task_id: int) -> str:
    return os.path.join("data/openml", str(task_id), "OpenMLTask.pkl")


def _split_by_fold(
    df: pd.DataFrame,
    fold_col: str,
    test_fold: int,
    n_valid_folds: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a DataFrame into train, valid, and test by fold.

    Args:
        df: The dataframe to split.
        fold_col: The name of the column containing the folds.
        test_fold: The fold to use for testing.
        n_valid_folds: The number of folds to use for validation.
    Returns:
        The train, valid, and test folds.
    """
    folds = [fold for fold in sorted(df[fold_col].unique()) if fold != test_fold]
    train_folds, valid_folds = folds[:-n_valid_folds], folds[-n_valid_folds:]
    train = df[df[fold_col].isin(train_folds)]
    valid = df[df[fold_col].isin(valid_folds)]
    test = df[df[fold_col] == test_fold]
    return train, valid, test


def dataset_from_task(
    task_id: int, test_fold: int, n_valid_folds: int = 3
) -> dataset.Dataset:
    """Creates a Dataset from an OpenMLTask.

    Args:
        task_id: The OpenML task id.
        test_fold: The fold to use for testing.
        n_valid_folds: The number of folds to use for validation. The folds are sorted
            and the largest n_validation_fols are used for validation.
    Returns:
        The dataset created from the task.
    """
    path = task_path(task_id)
    task = OpenMLTask.load(path)
    train, valid, test = _split_by_fold(
        task.df, task.fold_col, test_fold, n_valid_folds
    )
    all_cols = task.feature_cols + [task.label_col]
    return dataset.Dataset(
        train=train[all_cols].copy(),
        valid=valid[all_cols].copy(),
        test=test[all_cols].copy(),
        feature_cols=task.feature_cols,
        numerical_cols=task.numerical_cols,
        categorical_cols=task.categorical_cols,
        label_col=task.label_col,
    )
