from typing import Dict, Union

from automl import openml_utils
from automl import pipeline
from fastapi import FastAPI, Request
from pydantic import BaseModel
from ray import serve
import pandas as pd
import ray

app = FastAPI()


class CreateModelRequest(BaseModel):
    feature_schema: Dict[str, str]
    label_column: str


class CreateModelResponse(BaseModel):
    model_id: int


class PredictRequest(BaseModel):
    model_id: int
    features: Dict[str, Union[float, int, bool, str]]


class PredictResponse(BaseModel):
    prediction_id: int
    probs: Dict[str, float]


class TrainRequest(BaseModel):
    prediction_id: int
    label: Dict[str, str]


class TrainResponse(BaseModel):
    ...


@serve.deployment(route_prefix="/")
@serve.ingress(app)
class Server:
    def __init__(self):
        self.features_table = {}
        self.labels_table = {}
        self.model = None

    @app.get("/models/fit")
    async def test_fit_model(self):
        """Fits the model for testing purposes."""
        joined = []
        for prediction_id, features in self.features_table.items():
            label = self.labels_table[prediction_id]
            joined.append({**features, **label})
        train_df = pd.DataFrame.from_records(joined)
        self.model.fit(train_df)
        return {}

    @app.post("/models/create", response_model=CreateModelResponse)
    async def create_model(self, request: CreateModelRequest) -> CreateModelResponse:
        """Creates a new model with the given schema."""
        numerical_columns, categorical_columns = [], []
        for name, type_ in request.feature_schema.items():
            array = numerical_columns if type_ == "num" else categorical_columns
            array.append(name)
        self.model = pipeline.Pipeline(
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            label_column=request.label_column,
        )
        return CreateModelResponse(model_id=1)

    @app.post("/models/predict", response_model=PredictResponse)
    async def predict(self, request: PredictRequest) -> PredictResponse:
        """Stores the example and (possibly) returns model predictions."""
        # Store the observed features so we can later join them with the label.
        example = request.features
        prediction_id = len(self.features_table)
        self.features_table[prediction_id] = example

        # Model has not been trained yet. Return empty probs and leave it to the user
        # to handle this case.
        if not self.model.model:
            return PredictResponse(prediction_id=prediction_id, probs={})

        # Predict on the example if a model has been trained.
        df = pd.DataFrame.from_records([example])
        probs = self.model.predict(df)[self.model.prediction_column].values
        return PredictResponse(
            prediction_id=prediction_id,
            probs={
                self.model.classes[1]: probs[0],
                self.model.classes[0]: 1 - probs[0],
            },
        )

    @app.post("/models/train", response_model=TrainResponse)
    async def train(self, request: TrainRequest) -> TrainResponse:
        """Stores label for an observed example so they can be joined later."""
        # Store the observed label so we can later join it with the features.
        self.labels_table[request.prediction_id] = request.label
        return TrainResponse()
