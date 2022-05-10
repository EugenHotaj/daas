from typing import Dict, List, Union

import pandas as pd
import ray
from fastapi import FastAPI, Request
from pydantic import BaseModel
from ray import serve

from daas.automl import pipeline

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
        self.model_store = {}

    @app.get("/models/fit")
    async def test_fit_model(self, model_id: int):
        """Fits the model for testing purposes."""
        joined = []
        for prediction_id, features in self.features_table.items():
            label = self.labels_table[prediction_id]
            joined.append({**features, **label})
        train_df = pd.DataFrame.from_records(joined)
        self.model_store[model_id].fit(train_df)
        return {}

    @app.post("/models/create", response_model=CreateModelResponse)
    async def create_model(self, request: CreateModelRequest) -> CreateModelResponse:
        """Creates a new model with the given schema."""
        numerical_columns, categorical_columns = [], []
        for name, type_ in request.feature_schema.items():
            array = numerical_columns if type_ == "float" else categorical_columns
            array.append(name)
        model_id = len(self.model_store)
        self.model_store[model_id] = pipeline.Pipeline(
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            label_column=request.label_column,
        )
        return CreateModelResponse(model_id=model_id)

    @app.post("/models/predict", response_model=PredictResponse)
    async def predict(self, request: PredictRequest) -> PredictResponse:
        """Stores the example and (possibly) returns model predictions."""
        return await self._batch_predict(request)

    @serve.batch
    async def _batch_predict(
        self, requests: List[PredictRequest]
    ) -> List[PredictResponse]:
        responses = []
        for request in requests:
            # Store the observed features so we can later join them with the label.
            example = request.features
            prediction_id = len(self.features_table)
            self.features_table[prediction_id] = example

            model = self.model_store[request.model_id]
            if model.is_trained:
                # Predict on the example if a model has been trained.
                df = pd.DataFrame.from_records([example])
                # TODO(eugenhotaj): Actually batch the requests to the model.
                probs = model.predict(df)[model.prediction_column].values
                response = PredictResponse(
                    prediction_id=prediction_id,
                    probs={model.classes[1]: probs[0], model.classes[0]: 1 - probs[0]},
                )
            else:
                # Return empty probs when model has not yet been trained.
                response = PredictResponse(prediction_id=prediction_id, probs={})
            responses.append(response)
        return responses

    @app.post("/models/train", response_model=TrainResponse)
    async def train(self, request: TrainRequest) -> TrainResponse:
        """Stores label for an observed example so they can be joined later."""
        # Store the observed label so we can later join it with the features.
        self.labels_table[request.prediction_id] = request.label
        return TrainResponse()
