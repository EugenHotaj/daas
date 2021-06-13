import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import impute
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline

from automl import dataset


def categorical_encoder(df: pd.DataFrame) -> pipeline.Pipeline:
    imputer = impute.SimpleImputer(
        strategy="constant", fill_value="", add_indicator=True
    )
    ordinal_encoder = preprocessing.OrdinalEncoder(
        dtype=np.int32, handle_unknown="use_encoded_value", unknown_value=-1
    )
    encoder = pipeline.Pipeline(
        steps=[("imputer", imputer), ("ordinal_encoder", ordinal_encoder)]
    )
    encoder.fit(df)
    return encoder


def label_encoder(df: pd.DataFrame) -> preprocessing.LabelEncoder:
    encoder = preprocessing.LabelEncoder()
    encoder.fit(df)
    return encoder


def automl_pipeline(ds: dataset.Dataset, objective: str = "auc") -> lgb.Booster:
    # Preprocess features.
    categorical_cols = ds.categorical_cols
    if categorical_cols:
        encoder = categorical_encoder(ds.train[categorical_cols])

        encoded = encoder.transform(ds.train[categorical_cols])
        n_presence = encoded.shape[1] - len(categorical_cols)
        if n_presence > 0:
            encoded, _ = encoded[:, :-n_presence], encoded[:, -n_presence:]
        ds.train[categorical_cols] = encoded

        encoded = encoder.transform(ds.valid[categorical_cols])
        if n_presence > 0:
            encoded, _ = encoded[:, :-n_presence], encoded[:, -n_presence:]
        ds.valid[categorical_cols] = encoded

        encoded = encoder.transform(ds.test[categorical_cols])
        if n_presence > 0:
            encoded, _ = encoded[:, :-n_presence], encoded[:, -n_presence:]
        ds.test[categorical_cols] = encoded

    # Preprocess label.
    label_col = ds.label_col
    encoder = label_encoder(ds.train[label_col])
    ds.train[label_col] = encoder.transform(ds.train[label_col])
    ds.valid[label_col] = encoder.transform(ds.valid[label_col])
    ds.test[label_col] = encoder.transform(ds.test[label_col])

    feature_cols = ds.feature_cols
    train_data = lgb.Dataset(
        ds.train[feature_cols],
        label=ds.train[label_col],
        feature_name=feature_cols,
        categorical_feature=categorical_cols,
    )
    valid_data = lgb.Dataset(
        ds.valid[feature_cols],
        label=ds.valid[label_col],
        reference=train_data,
    )
    params = {
        "objective": "binary",
        "metric": [objective],
    }
    booster = lgb.train(
        params, train_data, 300, valid_sets=[valid_data], early_stopping_rounds=10
    )
    return booster
