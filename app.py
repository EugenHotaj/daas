from automl import dataset
from automl import openml_utils
from automl import pipeline
from download_data import BENCHMARK_TASKS
from fastapi import FastAPI
from ray import serve
import ray

app = FastAPI()


@serve.deployment(route_prefix="/")
@serve.ingress(app)
class Server:
    def __init__(self):
        ds = openml_utils.dataset_from_task(task_id=31, test_fold, n_valid_folds=2)
        model = pipeline.automl_pipeline(ds)

    @app.get("/predict")
    def increment(self):
        pass


ray.init(address="auto", namespace="serve")
Server.deploy()
