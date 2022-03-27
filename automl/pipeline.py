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
        self.processed_columns = None
        self.indicator_columns = None

        self._name = self.__class__.__name__

    @property
    def _indicator(self):
        # TODO(eugenhotaj): We're assuming the child class creates an imputer/indicator.
        return self.encoder["simple_imputer"].indicator_

    def fit(self, df: pd.DataFrame) -> None:
        self.encoder.fit(df[self.columns])
        self.processed_columns = [
            f"__{self._name}_processed_{col}__" for col in self.columns
        ]
        n_indicator_cols = self._indicator.features_.shape[0] if self._indicator else 0
        self.indicator_columns = [
            f"__{self._name}_indicator_{i}__" for i in range(n_indicator_cols)
        ]

    # TODO(ehotaj): Update transform to not modify df.
    def transform(self, df: pd.DataFrame) -> None:
        encoded, indicator = self.encoder.transform(df[self.columns]), None
        # TODO(ehotaj): It's much more efficient to work with sparse matricies.
        if scipy.sparse.issparse(encoded):
            encoded = encoded.todense()
        if self.indicator_columns:
            encoded = encoded[:, : -len(self.indicator_columns)]
            indicator = encoded[:, -len(self.indicator_columns) :]
            df[self.indicator_columnss] = indicator
        df[self.processed_columns] = encoded

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
        encoder = pipeline.Pipeline(
            steps=[
                ("label_encoder", self._label_encoder),
            ]
        )
        super().__init__(encoder, [column])

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
        self.model = None

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
        self.model = lgbm.train(params, train_data, model.best_iteration)

    def predict(self, df: pd.DataFrame) -> None:
        df[self.prediction_column] = self.model.predict(
            df[self.feature_columns], num_iteration=self.model.best_iteration
        )


class Pipeline:
    def __init__(
        self,
        numerical_columns: List[str],
        categorical_columns: List[str],
        label_column: str,
    ) -> None:
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.label_column = label_column

        # Create once self.fit() is called.
        self.numerical_encoder = None
        self.categorical_encoder = None
        self.label_encoder = None
        self.model = None
        self.prediction_column = None

        self._processed_feature_columns = []
        self._processed_label_column = None

    def _transform_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=False)  # Shallow copy because we don't modify original.
        self.numerical_encoder.transform(df)
        self.categorical_encoder.transform(df)
        self.label_encoder.transform(df)
        return df

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
        # Fit feature transforms.
        self.numerical_encoder = NumericalEncoder(columns=self.numerical_columns)
        self.numerical_encoder.fit(train_df)
        self._processed_feature_columns.extend(self.numerical_encoder.processed_columns)
        self._processed_feature_columns.extend(self.numerical_encoder.indicator_columns)

        self.categorical_encoder = CategoricalEncoder(columns=self.categorical_columns)
        self.categorical_encoder.fit(train_df)
        self._processed_feature_columns.extend(
            self.categorical_encoder.processed_columns
        )

        # Fit label transform.
        self.label_encoder = LabelEncoder(column=self.label_column)
        self.label_encoder.fit(train_df)
        self._processed_label_column = self.label_encoder.processed_columns[0]

        # Fit model.
        self.model = LightGBMModel(
            objective="binary",
            metric="auc",
            feature_columns=self._processed_feature_columns,
            label_column=self._processed_label_column,
        )
        train_df = self._transform_raw_features(train_df)
        valid_df = self._transform_raw_features(valid_df)
        self.model.fit(train_df, valid_df)
        self.prediction_column = self.model.prediction_column

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._transform_raw_features(df)  # Returns a shallow copy of df.
        self.model.predict(df)
        return df
