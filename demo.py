import requests

from app import Server
from daas.benchmark import openml_utils
import ray


@ray.remote
def send_example(model_id, features, label):
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

    print(f"prediction_id :: {prediction_id} probs :: {probs} label :: {label}")
    return prediction_id, probs, label["class"]


def send_examples(model_id, df, label_col):
    results = []
    for record in df.to_dict(orient="records"):
        record = {k: str(v) for k, v in record.items()}
        label = {label_col: record.pop(label_col)}
        result = send_example.remote(model_id, record, label)
        results.append(result)
    return ray.get(results)


def run_demo():
    dataset = openml_utils.dataset_from_task(31, 9)

    # Create model.
    request = {
        "feature_schema": {
            **{key: "float" for key in dataset.numerical_cols},
            **{key: "str" for key in dataset.categorical_cols},
        },
        "label_column": dataset.label_col,
    }
    response = requests.post("http://localhost:8000/models/create", json=request)
    response.raise_for_status()
    model_id = response.json()["model_id"]
    print(f"Model id :: {model_id}")

    # Train the model.
    send_examples(model_id, dataset.train, dataset.label_col)
    response = requests.get(f"http://localhost:8000/models/fit?model_id={model_id}")
    response.raise_for_status()

    # Make predictions.
    correct, total = 0, 0
    results = send_examples(model_id, dataset.test, dataset.label_col)
    for _, probs, label in results:
        best_pred, best_prob = None, 0.0
        for pred, prob in probs.items():
            if prob > best_prob:
                best_prob = prob
                best_pred = pred
        if best_pred == label:
            correct += 1
        total += 1
    print(f"Accuracy :: {correct / total:.2f}")


if __name__ == "__main__":
    ray.init(address="auto", namespace="serve")
    Server.deploy()
    run_demo()
