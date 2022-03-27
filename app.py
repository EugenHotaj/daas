from automl import openml_utils
from automl import pipeline
from fastapi import FastAPI
from ray import serve
import pandas as pd
import ray

app = FastAPI()


@serve.deployment(route_prefix="/")
@serve.ingress(app)
class Server:
    def __init__(self):
        self.dataset = openml_utils.dataset_from_task(
            task_id=31, test_fold=9, n_valid_folds=3
        )
        self.model = pipeline.Pipeline(
            numerical_columns=self.dataset.numerical_cols,
            categorical_columns=self.dataset.categorical_cols,
            label_column=self.dataset.label_col,
        )
        self.model.fit(self.dataset.train, self.dataset.valid)

    @app.get("/predict")
    def predict(self):
        example = {
            "checking_status": "<0",
            "duration": 6,
            "credit_history": "critical/other existing credit",
            "purpose": "radio/tv",
            "credit_amount": 1169,
            "savings_status": "no known savings",
            "employment": ">=7",
            "installment_commitment": 4,
            "personal_status": "male single",
            "other_parties": "none",
            "residence_since": 4,
            "property_magnitude": "real estate",
            "age": 67,
            "other_payment_plans": "none",
            "housing": "own",
            "existing_credits": 2,
            "job": "skilled",
            "num_dependents": 1,
            "own_telephone": "yes",
            "foreign_worker": "yes",
        }
        df = pd.DataFrame.from_records([example])
        probs = self.model.predict(df)[self.model.prediction_column].values
        return {"prob": {"good": probs[0], "bad": 1.0 - probs[0]}}


ray.init(address="auto", namespace="serve")
Server.deploy()
