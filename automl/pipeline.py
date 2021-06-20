from typing import Any, Dict, List, Tuple

import lightgbm as lgbm
import numpy as np
import pandas as pd
import scipy
from sklearn import impute
from sklearn import linear_model
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm

from automl import dataset

_MISSING = "__missing_value__"


class Encoder:
    """Base class for all encoders in the AutoML pipeline."""

    def __init__(
        self,
        encoder: pipeline.Pipeline,
        columns: List[str],
    ):
        """Initializes a new Encoder instance.

        Args:
            encoder: Encoder to use for transforming columns.
            columns: List of columns to encode.
        """
        self.encoder = encoder
        self.columns = columns
        self._name = self.__class__.__name__

        self.preprocessed_cols_ = None
        self.indicator_cols_ = None

    def fit(self, ds: dataset.Dataset) -> None:
        self.encoder.fit(ds.train[self.columns])
        self.preprocessed_cols_ = [
            f"__{self._name}_preprocessed_{col}__" for col in self.columns
        ]
        # TODO(eugenhotaj): This is pretty bad. We're assuming the child class creates
        # an imputer.
        indicator = self.encoder["simple_imputer"].indicator_
        n_indicator_cols = indicator.features_.shape[0] if indicator else 0
        self.indicator_cols_ = [
            f"__{self._name}_indicator_{i}__" for i in range(n_indicator_cols)
        ]

    def transform(self, ds: dataset.Dataset) -> None:
        for split in ("train", "valid", "test"):
            split = getattr(ds, split)
            encoded, indicator = self.encoder.transform(split[self.columns]), None
            # TODO(ehotaj): It's much more efficient to work with sparse matricies.
            if scipy.sparse.issparse(encoded):
                encoded = encoded.todense()
            if self.indicator_cols_:
                encoded = encoded[:, : -len(self.indicator_cols_)]
                indicator = encoded[:, -len(self.indicator_cols_) :]
                split[self.indicator_cols_] = indicator
            split[self.preprocessed_cols_] = encoded

    def fit_transform(self, ds: dataset.Dataset) -> None:
        self.fit(ds)
        self.transform(ds)


class CategoricalEncoder(Encoder):
    """Encodes arbitrary categorical variables into ints.

    Missing values are imputed to the empty string and (optionally) an indicator colum
    is added per column with missing values.
    """

    def __init__(self, columns: List[str]):
        self._simple_imputer = impute.SimpleImputer(
            strategy="constant", fill_value=_MISSING
        )
        self._ordinal_encoder = preprocessing.OrdinalEncoder(
            dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=-1
        )
        encoder = pipeline.Pipeline(
            steps=[
                ("simple_imputer", self._simple_imputer),
                ("ordinal_encoder", self._ordinal_encoder),
            ]
        )
        super().__init__(encoder, columns)


class OneHotEncoder(Encoder):
    """Encodes arbitrary categorical variables into one-hot numeric arrays.

    Missing values are imputed to the empty string and (optionally) an indicator colum
    is added per column with missing values.
    """

    def __init__(self, columns: List[str]):
        self._simple_imputer = impute.SimpleImputer(
            strategy="constant", fill_value=_MISSING
        )
        self._one_hot_encoder = preprocessing.OneHotEncoder(
            dtype=np.int32, handle_unknown="ignore"
        )
        encoder = pipeline.Pipeline(
            steps=[
                ("simple_imputer", self._simple_imputer),
                ("one_hot_encoder", self._one_hot_encoder),
            ]
        )
        super().__init__(encoder, columns)

    def fit(self, ds: dataset.Dataset) -> None:
        self.encoder.fit(ds.train[self.columns])
        self.indicator_cols_ = []
        cols = self.encoder["one_hot_encoder"].get_feature_names(self.columns)
        self.preprocessed_cols_ = [
            f"__{self._name}_preprocessed_{col}__" for col in cols
        ]


class NumericalEncoder(Encoder):
    """Normalizes numerical columns to have zero mean and unit variance.

    Missing values are imputed to the mean and (optionally) an indicator colum is added
    per column with missing values.
    """

    def __init__(self, columns: List[str]):
        self._simple_imputer = impute.SimpleImputer(strategy="mean", add_indicator=True)
        self._standard_scaler = preprocessing.StandardScaler()
        encoder = pipeline.Pipeline(
            steps=[
                ("simple_imputer", self._simple_imputer),
                ("standard_scaler", self._standard_scaler),
            ]
        )
        super().__init__(encoder, columns)


# TODO(eugenhotaj): LabelEncoder should not overwrite the raw label.
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


def _gather_cols(
    encoders: List[type], type_to_encoder: Dict[type, Encoder]
) -> Tuple[List[str], List[str]]:
    preprocessed_cols, indicator_cols = [], []
    for encoder in encoders:
        if encoder in type_to_encoder:
            preprocessed_cols += type_to_encoder[encoder].preprocessed_cols_
            indicator_cols += type_to_encoder[encoder].indicator_cols_
    return preprocessed_cols, indicator_cols


class LightGBMModel:
    def __init__(self, objective: str, metric: str):
        self.objective = objective
        self.metric = metric
        self.prediction_col = f"__{self.__class__.__name__}_predictions__"

        self.model_ = None
        self.feature_cols_ = None

    def fit(self, ds: dataset.Dataset, encoders: Dict[type, Encoder]) -> None:
        self.feature_cols_, categorical_cols = [], []
        cols, inds = _gather_cols([NumericalEncoder], encoders)
        self.feature_cols_ += cols + inds
        categorical_cols += inds

        cols, _ = _gather_cols([CategoricalEncoder], encoders)
        self.feature_cols_ += cols
        categorical_cols += cols

        train_data = lgbm.Dataset(
            ds.train[self.feature_cols_],
            label=ds.train[ds.label_col],
            feature_name=self.feature_cols_,
            categorical_feature=categorical_cols,
        )
        valid_data = lgbm.Dataset(
            ds.valid[self.feature_cols_],
            label=ds.valid[ds.label_col],
            reference=train_data,
        )
        params = {
            "objective": self.objective,
            "metric": [self.metric],
        }
        self.model_ = lgbm.train(
            params, train_data, 300, valid_sets=[valid_data], early_stopping_rounds=10
        )

    def predict(self, ds: dataset.Dataset) -> None:
        for split in ("train", "valid", "test"):
            split = getattr(ds, split)
            split[self.prediction_col] = self.model_.predict(
                split[self.feature_cols_], num_iteration=self.model_.best_iteration
            )


class LinearSVCModel:
    def __init__(self) -> None:
        self.prediction_col = f"__{self.__class__.__name__}_predictions__"
        self.model_ = None
        self.scaler_ = None
        self.feature_cols_ = None

    def fit(self, ds: dataset.Dataset, encoders: Dict[type, Encoder]) -> None:
        cols, inds = _gather_cols([NumericalEncoder, OneHotEncoder], encoders)
        self.feature_cols_ = cols + inds

        self.model_ = svm.LinearSVC(C=2 ** -7)
        self.model_.fit(ds.train[self.feature_cols_], ds.train[ds.label_col])

        # Platt scaling.
        self.scaler_ = linear_model.LogisticRegression(C=16.0)
        X = self.model_.decision_function(ds.valid[self.feature_cols_]).reshape(-1, 1)
        self.scaler_.fit(X, ds.valid[ds.label_col])

    def predict(self, ds: dataset.Dataset) -> None:
        for split in ("train", "valid", "test"):
            split = getattr(ds, split)
            X = self.model_.decision_function(split[self.feature_cols_]).reshape(-1, 1)
            split[self.prediction_col] = self.scaler_.predict_proba(X)[:, 1]


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
    type_to_encoder = {}
    # Preprocess features.
    if ds.numerical_cols:
        encoder = NumericalEncoder(columns=ds.numerical_cols)
        encoder.fit_transform(ds)
        type_to_encoder[NumericalEncoder] = encoder
    if ds.categorical_cols:
        encoder = CategoricalEncoder(columns=ds.categorical_cols)
        encoder.fit_transform(ds)
        type_to_encoder[CategoricalEncoder] = encoder

        encoder = OneHotEncoder(columns=ds.categorical_cols)
        encoder.fit_transform(ds)
        type_to_encoder[OneHotEncoder] = encoder

    # Preprocess label.
    LabelEncoder(column=ds.label_col).fit_transform(ds)

    # Train models.
    models = [LightGBMModel("binary", objective), LinearSVCModel()]
    for model in models:
        model.fit(ds, type_to_encoder)
        model.predict(ds)

    return model[0]
