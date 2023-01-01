"""Utilities for working with OpenML."""

from __future__ import annotations

import pathlib
import pickle
from dataclasses import dataclass
from typing import List

import pandas as pd

BENCHMARK_TASKS = {
    "kr-vs-kp": 3,
    "credit-g": 31,
    "kc1": 3917,
    "adult": 7592,
    "phoneme": 9952,
    "nomao": 9977,
    "bank-marketing": 14965,
    "higgs": 146606,
    "jasmine": 168911,
    "sylvine": 168912,
}
FOLD_COL = "fold"


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
            train=train[all_cols],
            test=test[all_cols],
            feature_cols=self.feature_cols,
            numerical_cols=self.numerical_cols,
            categorical_cols=self.categorical_cols,
            label_col=self.label_col,
        )


def task_path(task_id: int) -> str:
    # TODO(ehotaj): Do not hardcode data directory.
    dir_path = pathlib.Path(__file__).parent / "data" / "openml" / str(task_id)
    return dir_path / "OpenMLTask.pkl"


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
