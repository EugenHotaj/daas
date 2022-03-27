from automl import openml_utils
from automl import pipeline
from fastapi import FastAPI, Request
from ray import serve
import pandas as pd
import ray

app = FastAPI()


@serve.deployment(route_prefix="/")
@serve.ingress(app)
class Server:
    def __init__(self):
        self.features_table = {}
        self.labels_table = {}
        self.model = None

    @app.post("models/fit")
    async def fit_model(self):
        """Debug route which causes a new model to be trained on observed data."""
        self.model.fit(self.dataset.train, self.dataset.valid)

    @app.post("models/create")
    async def create_model(self, request: Request):
        """Creates a new model with the given schema."""
        request = await request.json()
        features = request["features"]
        numerical_columns, categorical_columns = [], []
        for name, type_ in features.items():
            if type_ in (bool, int, str):
                categorical_columns.append(name)
            else:
                numerical_columns.append(name)
        label_column = request["label"]
        self.model = pipeline.Pipeline(
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            label_column=label_column,
        )
        return {"id": 1}

    @app.post("models/predict")
    async def predict(self, request: Request):
        """Stores the example and (possibly) returns model predictions."""
        request = await request.json()

        # Store the observed features so we can later join them with the label.
        example = request["features"]
        prediction_id = len(self.features_table)
        self.features_table[prediction_id] = example

        # Model has not been trained yet. Return empty probs and leave it to the user
        # to handle this case.
        if not self.model.model:
            return {"prediction_id": prediction_id, "probs": {}}

        # Predict on the example if a model has been trained.
        df = pd.DataFrame.from_records([example])
        probs = self.model.predict(df)[self.model.prediction_column].values
        return {
            "prediction_id": prediction_id,
            # TODO(ehotaj): Do not hardcode the credit-g label here.
            "probs": {"good": probs[0], "bad": 1.0 - probs[0]},
        }

    @app.post("models/train")
    async def train(self, request: Request):
        """Stores label for an observed example so they can be joined later."""
        request = await request.json()

        # Store the observed label so we can later join it with the features.
        prediction_id = request["prediction_id"]
        label = request["label"]
        self.labels_table[prediction_id] = label
        return {}


ray.init(address="auto", namespace="serve")
Server.deploy()
