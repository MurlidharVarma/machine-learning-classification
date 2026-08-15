"""
Data related: Loading, encoding, splitting, and validation of uploaded files
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    FEATURES,
    GENDER_FEATURE,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    SEED,
    SYMPTOM_FEATURES,
    TARGET,
    TEST_SIZE,
)

YES_NO_MAP = {"No": 0, "Yes": 1}
GENDER_MAP = {"Female": 0, "Male": 1}
CLASS_MAP = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}

"""Read the dataset with no encoding applied."""
def load_data(source=DATA_PATH):
    return pd.read_csv(source)

"""
Map every categorical column to 0/1, leaving Age untouched.
"""
def encode_features(frame):

    encoded = frame.copy()

    for column in SYMPTOM_FEATURES:
        encoded[column] = encoded[column].map(YES_NO_MAP)

    encoded[GENDER_FEATURE] = encoded[GENDER_FEATURE].map(GENDER_MAP)

    if TARGET in encoded.columns:
        encoded[TARGET] = encoded[TARGET].map(CLASS_MAP)

    return encoded
