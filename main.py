import argparse
import collections
from typing import Any, Dict, Tuple

import numpy as np
import ray
from sklearn import metrics

from automl import dataset
from automl import openml_utils
from automl import pipeline
from download_data import BENCHMARK_TASKS


@ray.remote
def one_fold(task_id: int, fold: int) -> Dict[str, float]:
    ds = openml_utils.dataset_from_task(task_id, fold, n_valid_folds=2)
    model = pipeline.automl_pipeline(ds, "auc")
    labels = ds.test[ds.label_col]
    predictions = model.predict(
        ds.test[ds.feature_cols], num_iteration=model.best_iteration
    )
    metric = {
        "auc": metrics.roc_auc_score(labels, predictions),
        "pr_auc": metrics.average_precision_score(labels, predictions),
        "accuracy": metrics.accuracy_score(labels, np.where(predictions >= 0.5, 1, 0)),
    }
    return metric


@ray.remote
def run_on_task(task_id: int) -> Dict[str, Tuple[float, float]]:
    futures = [one_fold.remote(task_id, i) for i in range(10)]
    metric = collections.defaultdict(list)
    for fold in ray.get(futures):
        for key, value in fold.items():
            metric[key].append(value)
    return {key: (np.mean(value), np.std(value)) for key, value in metric.items()}


def print_result(task_id: int, task_name: str, results: Dict[str, float]):
    print(f"Results for task {task_id} [{task_name}]:")
    for key, value in results.items():
        mean, std = value
        print(f"  {key}: {round(mean, 3)} +/- {round(std, 3)}")
    print()


def main(args):
    ray.init(local_mode=args.local_mode)
    futures = {k: run_on_task.remote(v) for k, v in BENCHMARK_TASKS.items()}
    for task_name, result in zip(futures.keys(), ray.get(list(futures.values()))):
        print_result(BENCHMARK_TASKS[task_name], task_name, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-mode", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
