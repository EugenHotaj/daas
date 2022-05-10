from typing import Any, Dict, List, Optional, Tuple, Union

import lightgbm as lgbm
import numpy as np
import pandas as pd
import scipy
from sklearn import impute, pipeline, preprocessing

TDtype = Union[str, type, np.dtype]

# TODO(ehotaj): The distinction between Encoder/Model is pretty flimsy. Consider just
# having one Transform base class for everything.
class Encoder:
    """Base class for all encoders in the AutoML pipeline."""

    def __init__(
        self,
        encoder: pipeline.Pipeline,
        in_dtype: TDtype,
        out_dtype: TDtype,
        columns: List[str],
    ):
        """Initializes a new Encoder instance.

        Args:
            encoder: Encoder to use for transforming columns.
            in_dtype: The dtype to cast inputs to before encoding.
            out_dtype: The dtype to cast outputs to after encoding.
            columns: List of columns to encode.
        """
        self.encoder = encoder
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.columns = columns

        self.processed_columns = []
        self.indicator_columns = []

        self._name = self.__class__.__name__

    @property
    def _indicator(self):
        # TODO(eugenhotaj): We're assuming the child class creates an imputer/indicator.
        return self.encoder["simple_imputer"].indicator_

    def fit(self, df: pd.DataFrame) -> None:
        self.encoder.fit(df[self.columns].astype(self.in_dtype))
        self.processed_columns = [
            f"__{self._name}_processed_{col}__" for col in self.columns
        ]
        n_indicator_cols = self._indicator.features_.shape[0] if self._indicator else 0
        self.indicator_columns = [
            f"__{self._name}_indicator_{i}__" for i in range(n_indicator_cols)
        ]

    # TODO(ehotaj): Update transform to not modify df.
    def transform(self, df: pd.DataFrame) -> None:
        encoded = self.encoder.transform(df[self.columns].astype(self.in_dtype))
        # TODO(ehotaj): It's much more efficient to work with sparse matricies.
        if scipy.sparse.issparse(encoded):
            encoded = encoded.todense()
        if self.indicator_columns:
            df[self.indicator_columns] = encoded[:, -len(self.indicator_columns) :]
            # NOTE: Indicator columns should always be of type int64.
            df[self.indicator_columns] = df[self.indicator_columns].astype(np.int64)
            encoded = encoded[:, : -len(self.indicator_columns)]
        df[self.processed_columns] = encoded
        df[self.processed_columns] = df[self.processed_columns].astype(self.out_dtype)

    def fit_transform(self, df: pd.DataFrame) -> None:
        self.fit(df)
        self.transform(df)


class CategoricalEncoder(Encoder):
    """Encodes arbitrary categorical variables into ints.

    Missing values are imputed to the empty string and (optionally) an indicator colum
    is added per column with missing values.
    """

    MISSING_VALUE = "__MISSING__"

    def __init__(self, columns: List[str]):
        self._simple_imputer = impute.SimpleImputer(
            strategy="constant",
            fill_value=CategoricalEncoder.MISSING_VALUE,
            add_indicator=True,
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
        super().__init__(
            encoder=encoder, in_dtype=str, out_dtype="category", columns=columns
        )


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
        super().__init__(
            encoder=encoder, in_dtype=np.float64, out_dtype=np.float64, columns=columns
        )


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
        super().__init__(
            encoder=encoder, in_dtype=str, out_dtype=np.int64, columns=[column]
        )

        # Set after fit() is called.
        self.classes = None

    def fit(self, df: pd.DataFrame) -> None:
        super().fit(df)
        self.classes = self._label_encoder.categories_[0].tolist()

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

        # Set after fit() is called.
        self.cv_booster = None
        self.full_booster = None
        self.best_iteration = None

    def fit(self, df: pd.DataFrame) -> None:
        # TODO(ehotaj): Use a more principled approach to stabalize small datasets.
        # Some possibilities: (1) less complex model, (2) tune hparams.
        if len(df) < 10000:
            replicates = int(np.ceil(10000 / len(df)))
            df = pd.concat([df] * replicates)

        # Early stopping.
        train_set = lgbm.Dataset(df[self.feature_columns], label=df[self.label_column])
        params = {
            "objective": self.objective,
            "metric": [self.metric],
            "num_boost_round": 500,
            "early_stopping_rounds": 50,
        }
        result = lgbm.cv(params=params, train_set=train_set, return_cvbooster=True)
        self.cv_booster = result["cvbooster"]
        self.best_iteration = self.cv_booster.best_iteration

        # Full dataset.
        del params["early_stopping_rounds"]
        del params["metric"]
        params["num_boost_round"] = self.best_iteration
        self.full_booster = lgbm.train(params=params, train_set=train_set)

    def predict(self, df: pd.DataFrame) -> None:
        features = df[self.feature_columns]
        cv_preds = np.mean(self.cv_booster.predict(features), axis=0)
        full_preds = self.full_booster.predict(features)
        df[self.prediction_column] = 0.5 * cv_preds + 0.5 * full_preds


class Pipeline:
    """The AutoML pipeline.

    Represents a (trainable) function fn(raw_inputs)->raw_preds. This class encapsulates
    all parts of the machine learning pipeline (e.g. feature transforms, model(s),
    ensemble, etc) and ensures that the training and inference path are the same.
    """

    def __init__(
        self,
        numerical_columns: List[str],
        categorical_columns: List[str],
        label_column: str,
    ) -> None:
        """Initializes a new Pipeline instance.

        Args:
            numerical_columns: Names of continuous valued columns.
            categorical_columns: Names of categorical columns.
            label_column: Name of the label column.
        """
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.label_column = label_column

        self.is_trained = False

        # Create once self.fit() is called.
        self.numerical_encoder = None
        self.categorical_encoder = None
        self.label_encoder = None
        self.model = None
        self.prediction_column = None
        self.classes = None

        self._processed_feature_columns = []
        self._processed_label_column = None

    def _transform_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=False)  # Shallow copy because we don't modify original.
        if self.numerical_columns:
            self.numerical_encoder.transform(df)
        if self.categorical_columns:
            self.categorical_encoder.transform(df)
        if self.label_column in df.columns:
            self.label_encoder.transform(df)
        return df

    def fit(self, df: pd.DataFrame) -> None:
        """Fits the whole AutoML pipeline on the given train set."""
        # TODO(ehotaj): Shuffle dataset... should not make a difference but somehow
        # it boosts performance for some datsets?
        df = df.iloc[np.random.permutation(len(df))]

        # Fit feature transforms.
        if self.numerical_columns:
            self.numerical_encoder = NumericalEncoder(columns=self.numerical_columns)
            self.numerical_encoder.fit(df)
            self._processed_feature_columns.extend(
                self.numerical_encoder.processed_columns
            )
            self._processed_feature_columns.extend(
                self.numerical_encoder.indicator_columns
            )

        if self.categorical_columns:
            self.categorical_encoder = CategoricalEncoder(
                columns=self.categorical_columns
            )
            self.categorical_encoder.fit(df)
            self._processed_feature_columns.extend(
                self.categorical_encoder.processed_columns
            )
            self._processed_feature_columns.extend(
                self.categorical_encoder.indicator_columns
            )

        # Fit label transform.
        self.label_encoder = LabelEncoder(column=self.label_column)
        self.label_encoder.fit(df)
        self.classes = self.label_encoder.classes
        self._processed_label_column = self.label_encoder.processed_columns[0]

        # Fit model.
        self.model = LightGBMModel(
            objective="binary",
            metric="auc",
            feature_columns=self._processed_feature_columns,
            label_column=self._processed_label_column,
        )
        df = self._transform_raw_features(df)
        self.model.fit(df)
        self.prediction_column = self.model.prediction_column
        self.is_trained = True

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a copy of the dataframe with predictions."""
        df = self._transform_raw_features(df)  # Return shallow copy.
        self.model.predict(df)
        return df
