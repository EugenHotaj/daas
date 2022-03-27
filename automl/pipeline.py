from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgbm
import numpy as np
import pandas as pd
import scipy
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing

from automl import dataset

_MISSING = "__missing_value__"


# TODO(ehotaj): The distinction between Encoder/Model is pretty flimsy. Consider just
# having one Transform base class for everything.
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

    @property
    def _indicator(self):
        # TODO(eugenhotaj): We're assuming the child class creates an imputer/indicator.
        return self.encoder["simple_imputer"].indicator_

    def fit(self, df: pd.DataFrame) -> None:
        self.encoder.fit(df[self.columns])
        self.preprocessed_cols_ = [
            f"__{self._name}_preprocessed_{col}__" for col in self.columns
        ]
        n_indicator_cols = self._indicator.features_.shape[0] if self._indicator else 0
        self.indicator_cols_ = [
            f"__{self._name}_indicator_{i}__" for i in range(n_indicator_cols)
        ]

    def transform(self, df: pd.DataFrame) -> None:
        encoded, indicator = self.encoder.transform(df[self.columns]), None
        # TODO(ehotaj): It's much more efficient to work with sparse matricies.
        if scipy.sparse.issparse(encoded):
            encoded = encoded.todense()
        if self.indicator_cols_:
            encoded = encoded[:, : -len(self.indicator_cols_)]
            indicator = encoded[:, -len(self.indicator_cols_) :]
            df[self.indicator_cols_] = indicator
        df[self.preprocessed_cols_] = encoded

    def fit_transform(self, df: pd.DataFrame) -> None:
        self.fit(df)
        self.transform(df)


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
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )
        encoder = pipeline.Pipeline(
            steps=[
                ("simple_imputer", self._simple_imputer),
                ("ordinal_encoder", self._ordinal_encoder),
            ]
        )
        super().__init__(encoder, columns)


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


class LabelEncoder(Encoder):
    """Encodes a label colum into ints."""

    def __init__(self, column: str):
        """Initializes a new LabelEncoder instance.

        Args:
            column: The label column to encode.
        """

        # NOTE: We use OrdinalEncoder because LabelEncoder does not work with the
        # SKLearn pipeline interface.
        self._label_encoder = preprocessing.OrdinalEncoder()
        self.encoder = pipeline.Pipeline(
            steps=[
                ("label_encoder", self._label_encoder),
            ]
        )
        super().__init__(self.encoder, [column])

    @property
    def _indicator(self):
        return None


class LightGBMModel:
    def __init__(
        self, objective: str, metric: str, feature_columns: List[str], label_column: str
    ):
        self.objective = objective
        self.metric = metric
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.prediction_column = f"__{self.__class__.__name__}_predictions__"

        self.model_ = None

    def _make_dataset(self, df: pd.DataFrame, reference: Optional[lgbm.Dataset] = None):
        # TODO(ehotaj): Should we explicitly specify categorical_feature here?
        return lgbm.Dataset(
            df[self.feature_columns],
            label=df[self.label_column],
            feature_name=self.feature_columns,
            reference=reference,
        )

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
        # Early stopping.
        train_data = self._make_dataset(train_df)
        valid_data = self._make_dataset(valid_df, reference=train_data)
        params = {
            "objective": self.objective,
            "metric": [self.metric],
        }
        model = lgbm.train(
            params, train_data, 500, valid_sets=[valid_data], early_stopping_rounds=50
        )

        # Full dataset.
        train_data = self._make_dataset(pd.concat([train_df, valid_df]))
        params = {
            "objective": self.objective,
        }
        self.model_ = lgbm.train(params, train_data, model.best_iteration)

    def predict(self, df: pd.DataFrame) -> None:
        df[self.prediction_column] = self.model_.predict(
            df[self.feature_columns], num_iteration=self.model_.best_iteration
        )


def _fit_and_apply(
    encoder_or_model: Union[Encoder, LightGBMModel], ds: dataset.Dataset
) -> None:
    try:
        encoder_or_model.fit(ds.train)
    except TypeError as e:
        # Some fit() functions expect a validation dataset.
        if not "required positional argument" in str(e):
            raise
        encoder_or_model.fit(ds.train, ds.valid)
    apply_fn = (
        encoder_or_model.transform
        if hasattr(encoder_or_model, "transform")
        else encoder_or_model.predict
    )
    apply_fn(ds.train)
    apply_fn(ds.valid)
    apply_fn(ds.test)


# TODO(eugenhotaj): The pipeline should not just return the model, but rather a
# predictor which takes as input raw data and produces predictions.
def automl_pipeline(ds: dataset.Dataset) -> LightGBMModel:
    """Entry point for the AutoML pipeline.

    Args:
        ds: The Dataset to use for training and evaluating the AutoML pipeline.
    Returns:
        The best model found.
    """
    # Preprocess features.
    feature_cols = []
    if ds.numerical_cols:
        encoder = NumericalEncoder(columns=ds.numerical_cols)
        _fit_and_apply(encoder, ds)
        feature_cols.extend(encoder.preprocessed_cols_)
        feature_cols.extend(encoder.indicator_cols_)
    if ds.categorical_cols:
        encoder = CategoricalEncoder(columns=ds.categorical_cols)
        _fit_and_apply(encoder, ds)
        feature_cols.extend(encoder.preprocessed_cols_)

    # Preprocess label.
    encoder = LabelEncoder(column=ds.label_col)
    _fit_and_apply(encoder, ds)
    label_col = encoder.preprocessed_cols_[0]

    # Train models.
    model = LightGBMModel(
        objective="binary",
        metric="auc",
        feature_columns=feature_cols,
        label_column=label_col,
    )
    _fit_and_apply(model, ds)
    return model
