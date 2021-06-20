from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class Dataset:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    feature_cols: List[str]
    numerical_cols: List[str]
    categorical_cols: List[str]
    label_col: str
