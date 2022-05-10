import argparse

import ray
from daas.benchmark.core import run
from daas.benchmark.openml_utils import BENCHMARK_TASKS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--dataset", choices=BENCHMARK_TASKS.keys(), default=None)
    args = parser.parse_args()

    ray.init(address="auto", local_mode=args.local)
    run(args.dataset)
