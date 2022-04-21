import requests

from app import Server
from automl import openml_utils
import pandas as pd
import ray


def send_example(model_id, features, label=None):
    # Make a prediction.
    request = {"model_id": model_id, "features": features}
    response = requests.post("http://localhost:8000/models/predict", json=request)
    response.raise_for_status()
    response = response.json()
    prediction_id = response["prediction_id"]
    probs = response["probs"]

    # Observe the label if provided.
    if label:
        request = {"prediction_id": prediction_id, "label": label}
        response = requests.post("http://localhost:8000/models/train", json=request)
        response.raise_for_status()

    return prediction_id, probs


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
    response.raise_for_status()
    model_id = response.json()["model_id"]
    print(f"Model id :: {model_id}")

    # Make some predictions and observe the label.
    records = train.to_dict(orient="records")
    for record in records[:10]:
        label = {dataset.label_col: record.pop(dataset.label_col)}
        prediction_id, probs = send_example(model_id, record, label)
        print(f"prediction_id :: {prediction_id} probs :: {probs}")

    # Train the model.
    response = requests.get("http://localhost:8000/models/fit")
    response.raise_for_status()

    # Make some more predictions.
    for record in records[10:15]:
        record.pop(dataset.label_col)  # Remove label.
        prediction_id, probs = send_example(model_id, record)
        print(f"prediction_id :: {prediction_id} probs :: {probs}")


if __name__ == "__main__":
    ray.init(address="auto", namespace="serve")
    Server.deploy()
    run_demo()
