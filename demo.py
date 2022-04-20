import requests

from app import Server
from automl import openml_utils
import pandas as pd
import ray


def run_demo():
    dataset = openml_utils.dataset_from_task(31, 9, n_valid_folds=1)
    train, test = pd.concat([dataset.train, dataset.valid]), dataset.test

    # Create model.
    request = {
        "feature_schema": {
            **{key: "num" for key in dataset.numerical_cols},
            **{key: "cat" for key in dataset.categorical_cols},
        },
        "label_column": dataset.label_col,
    }
    response = requests.post("http://localhost:8000/models/create", json=request)
    model_id = response.json()["model_id"]
    print(f"Model id :: {model_id}")

    # Make some predictions and observe the label.
    for record in train.to_dict(orient="records")[:10]:
        # Make a prediction.
        label = record.pop(dataset.label_col)
        request = {"model_id": model_id, "features": record}
        response = requests.post("http://localhost:8000/models/predict", json=request)
        response = response.json()
        prediction_id = response["prediction_id"]
        print(f"prediction_id :: {prediction_id}")
        print(f"probs :: {response['probs']}")  # Should be empty.

        # Observe the label.
        request = {"prediction_id": predictino_id, "label": {dataset.label_col: label}}
        response = requests.post("http://localhost:8000/models/train", json=request)
        print(response)


if __name__ == "__main__":
    ray.init(address="auto", namespace="serve")
    Server.deploy()
    run_demo()
