"""Script to download and cache all data."""

import os
from typing import List

import openml

from automl import openml_utils

BENCHMARK_TASKS = {"credit-g": 31, "adult": 7592, "nomao": 9977, "phoneme": 9952}
FOLD_COL = "fold"


def download_openml_tasks(task_ids: List[int]):
    """Downloads the given task_ids from OpenML and dumps them as OpenMLTasks."""
    tasks = openml.tasks.get_tasks(
        task_ids, download_data=True, download_qualities=False
    )
    for task in tasks:
        dataset = task.get_dataset()
        df, _, categorical, columns = dataset.get_data()
        label_col = dataset.default_target_attribute
        feature_cols = [col for col in columns if col != label_col]
        numerical_cols = [col for ind, col in zip(categorical, feature_cols) if not ind]
        categorical_cols = [col for ind, col in zip(categorical, feature_cols) if ind]

        df[FOLD_COL] = -1
        splits = task.download_split().split[0]  # We assume one repetition.
        for split, idxs in splits.items():
            idxs = idxs[0].test
            df.loc[idxs, FOLD_COL] = split

        out_path = openml_utils.task_path(task.task_id)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        task = openml_utils.OpenMLTask(
            df, feature_cols, numerical_cols, categorical_cols, label_col, FOLD_COL
        )
        task.dump(out_path)


if __name__ == "__main__":
    download_openml_tasks(list(BENCHMARK_TASKS.values()))
