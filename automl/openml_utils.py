"""Utilities for working with OpenML."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
import os
import numpy as np
from typing import List

import pandas as pd


@dataclass
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    feature_cols: List[str]
    numerical_cols: List[str]
    categorical_cols: List[str]
    label_col: str


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

    def create_dataset(self, test_fold: int) -> Dataset:
        """Creates a Dataset from the OpenML task.

        Args:
            test_fold: Fold to use for testing.
        Returns:
            Dataset created from the task.
        """
        folds = [f for f in sorted(self.df[self.fold_col].unique()) if f != test_fold]
        train = self.df[self.df[self.fold_col].isin(folds)]
        test = self.df[self.df[self.fold_col] == test_fold]
        all_cols = self.feature_cols + [self.label_col]
        return Dataset(
            train=train[all_cols].copy(),
            test=test[all_cols].copy(),
            feature_cols=self.feature_cols,
            numerical_cols=self.numerical_cols,
            categorical_cols=self.categorical_cols,
            label_col=self.label_col,
        )


def task_path(task_id: int) -> str:
    return os.path.join("data/openml", str(task_id), "OpenMLTask.pkl")


def dataset_from_task(task_id: int, test_fold: int) -> Dataset:
    """Creates a Dataset from an OpenMLTask.

    Args:
        task_id: OpenML task id.
        test_fold: Fold to use for testing.
    Returns:
        Dataset created from the task.
    """
    path = task_path(task_id)
    return OpenMLTask.load(path).create_dataset(test_fold)
