import argparse
import collections
from typing import Any, Dict, Optional, Tuple

import numpy as np
import ray
from sklearn import metrics as sklearn_metrics

from automl import dataset
from automl import openml_utils
from automl import pipeline
from download_data import BENCHMARK_TASKS


@ray.remote
def one_fold(task_id: int, test_fold: int) -> Dict[str, Dict[str, float]]:
    ds = openml_utils.dataset_from_task(task_id, test_fold, n_valid_folds=2)
    model = pipeline.automl_pipeline(ds)
    metrics = {}
    for split in ("train", "valid", "test"):
        df = getattr(ds, split)
        predictions = df[model.prediction_column]
        labels = df[model.label_column]
        metric = {
            "auc": sklearn_metrics.roc_auc_score(labels, predictions),
            "pr_auc": sklearn_metrics.average_precision_score(labels, predictions),
            "accuracy": sklearn_metrics.accuracy_score(
                labels, np.where(predictions >= 0.5, 1, 0)
            ),
        }
        metrics[split] = metric
    return metrics


TMetrics = Dict[str, Dict[str, Tuple[float, float]]]


@ray.remote
def run_on_task(task_id: int) -> TMetrics:
    futures = [one_fold.remote(task_id, i) for i in range(10)]
    metrics = collections.defaultdict(lambda: collections.defaultdict(list))
    for fold in ray.get(futures):
        for split, split_metrics in fold.items():
            for metric, value in split_metrics.items():
                metrics[split][metric].append(value)
    for split, split_metrics in metrics.items():
        metrics[split] = {
            key: (np.mean(value), np.std(value)) for key, value in split_metrics.items()
        }
    return dict(metrics)


def print_metrics(task_id: int, task_name: str, metrics: TMetrics) -> None:
    print(f"Metrics for task {task_id} [{task_name}]:")
    metric_to_split = collections.defaultdict(dict)
    for split, split_metrics in metrics.items():
        for metric, value in split_metrics.items():
            mean, std = value
            metric_to_split[metric][split] = (mean, mean - std, mean + std)
    for metric, split in metric_to_split.items():
        print(f"  {metric}: ")
        for split, values in split.items():
            mean, lo, hi = values
            print(f"    {split}: {round(mean, 4)} ({round(lo, 4)} {round(hi, 4)})")
    print()


def main(args):
    ray.init(local_mode=args.local)
    if args.dataset is None:
        futures = {k: run_on_task.remote(v) for k, v in BENCHMARK_TASKS.items()}
    else:
        futures = {args.dataset: run_on_task.remote(BENCHMARK_TASKS[args.dataset])}
    for task_name, result in zip(futures.keys(), ray.get(list(futures.values()))):
        print_metrics(BENCHMARK_TASKS[task_name], task_name, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--dataset", choices=BENCHMARK_TASKS.keys(), default=None)
    args = parser.parse_args()
    main(args)
