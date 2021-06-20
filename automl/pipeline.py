from typing import Any, List

import lightgbm as lgbm
import numpy as np
import pandas as pd
from sklearn import base
from sklearn import impute
from sklearn import metrics
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm

from automl import dataset


# TODO(eugenhotaj): Extract an Encoder interface.
class CategoricalEncoder:
    """Encodes arbitrary categorical variables into ints.

    Missing values are imputed to the empty string and (optionally) an indicator colum
    is added per column with missing values.
    """

    def __init__(self, columns: List[str], add_indicator: bool = False):
        """Initializes a new CategoricalEncoder instance.

        Args:
            columns: The list of categorical columns to encode.
            add_indicator: If True, an indicator column will be atted to the DataFrames
                for each column which contains missing values.
        """
        self.columns = columns
        self.add_indicator = add_indicator
        self.n_missing_cols = None
        self._imputer = impute.SimpleImputer(
            strategy="constant", fill_value="", add_indicator=add_indicator
        )
        self._ordinal_encoder = preprocessing.OrdinalEncoder(
            dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.encoder = pipeline.Pipeline(
            steps=[
                ("imputer", self._imputer),
                ("ordinal_encoder", self._ordinal_encoder),
            ]
        )

    def fit(self, ds: dataset.Dataset):
        self.encoder.fit(ds.train[self.columns])
        if self.add_indicator:
            self.n_missing_cols = self._imputer.indicator_.features_.shape[1]

    def transform(self, ds: dataset.Dataset):
        for split in ("train", "valid", "test"):
            split = getattr(ds, split)
            encoded = self.encoder.transform(split[self.columns])
            if self.n_missing_cols:
                encoded = encoded[:, : -self.n_missing_cols]
                missing = encoded[:, -self.n_missing_cols :]
            split[self.columns] = encoded
            if self.add_indicator and self.n_missing_cols:
                # TODO(eugenhotaj): Handle missing indicator.
                pass

    def fit_transform(self, ds: dataset.Dataset):
        self.fit(ds)
        self.transform(ds)


class LabelEncoder:
    """Encodes a label colum into ints."""

    def __init__(self, column: str):
        """Initializes a new LabelEncoder instance.

        Args:
            column: The label column to encode.
        """
        self.column = column
        self.encoder = preprocessing.LabelEncoder()

    def fit(self, ds: dataset.Dataset):
        self.encoder.fit(ds.train[self.column])

    def transform(self, ds: dataset.Dataset):
        for split in ("train", "valid", "test"):
            split = getattr(ds, split)
            split[self.column] = self.encoder.transform(split[self.column])

    def fit_transform(self, ds: dataset.Dataset):
        self.fit(ds)
        self.transform(ds)


class LightGBMClassifier(base.ClassifierMixin):
    def __init__(self, model: lgbm.Booster):
        self.model = booster

    # TODO(eugenhotaj): Add type information.
    def predict(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration)


# TODO(eugenhotaj): model should not be a string but a search space.
def train_model(ds: dataset.Dataset, model="lgbm", objective: str = "auc") -> Any:
    if model == "lgbm":
        # Train LightGBM.
        train_data = lgbm.Dataset(
            ds.train[ds.feature_cols],
            label=ds.train[ds.label_col],
            feature_name=ds.feature_cols,
            categorical_feature=ds.categorical_cols,
        )
        valid_data = lgbm.Dataset(
            ds.valid[ds.feature_cols],
            label=ds.valid[ds.label_col],
            reference=train_data,
        )
        params = {
            "objective": "binary",
            "metric": [objective],
        }
        model = lgbm.train(
            params, train_data, 300, valid_sets=[valid_data], early_stopping_rounds=10
        )
        model = LightGBMClassifier(model)
    elif model == "svm":
        model = svm.LinearSVC()
        model.fit(ds.train[ds.feature_cols], ds.train[ds.label_col])
    else:
        raise ValueError("Unknown model :: {model}.")
    return model


# TODO(eugenhotaj): The pipeline should not just return the model, but rather a
# predictor which takes as input raw data and produces predictions.
def automl_pipeline(ds: dataset.Dataset, objective: str = "auc") -> Any:
    """Entry point for the AutoML pipeline.

    Args:
        ds: The Dataset to use for training and evaluating the AutoML pipeline.
        objective: The main metric to optimize. Does not need to be differentiable.
    Returns:
        The best model found.
    """
    # Preprocess features.
    if ds.categorical_cols:
        CategoricalEncoder(columns=ds.categorical_cols).fit_transform(ds)
    # Preprocess label.
    LabelEncoder(column=ds.label_col).fit_transform(ds)
    model = train_model(ds=ds, model="lgbm", objective=objective)
    return model
