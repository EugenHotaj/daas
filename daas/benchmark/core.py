import collections
from typing import Dict, Optional, Tuple

import numpy as np
import ray
from sklearn import metrics as sklearn_metrics

from daas.automl import pipeline as automl_pipeline
from daas.benchmark.openml_utils import BENCHMARK_TASKS, dataset_from_task

TMetrics = Dict[str, Dict[str, Tuple[float, float]]]


@ray.remote
def one_fold(task_id: int, test_fold: int) -> Dict[str, Dict[str, float]]:
    dataset = dataset_from_task(task_id, test_fold)
    pipeline = automl_pipeline.Pipeline(
        numerical_columns=dataset.numerical_cols,
        categorical_columns=dataset.categorical_cols,
        label_column=dataset.label_col,
    )
    pipeline.fit(dataset.train)
    metrics = {}
    for split, split_df in [("train", dataset.train), ("test", dataset.test)]:
        pred_df = pipeline.predict(split_df)
        predictions = pred_df[pipeline.prediction_column]
        # TODO(ehotaj): Expose _processed_label_column?
        labels = pred_df[pipeline._processed_label_column]
        metric = {
            "auc": sklearn_metrics.roc_auc_score(labels, predictions),
            "pr_auc": sklearn_metrics.average_precision_score(labels, predictions),
            "accuracy": sklearn_metrics.accuracy_score(
                labels, np.where(predictions >= 0.5, 1, 0)
            ),
            "best_iteration": pipeline.model.best_iteration,
        }
        metrics[split] = metric
    return metrics


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


def run(dataset: Optional[str] = None) -> None:
    if dataset:
        futures = {dataset: run_on_task.remote(BENCHMARK_TASKS[dataset])}
    else:
        futures = {k: run_on_task.remote(v) for k, v in BENCHMARK_TASKS.items()}
    for task_name, result in zip(futures.keys(), ray.get(list(futures.values()))):
        print_metrics(BENCHMARK_TASKS[task_name], task_name, result)
