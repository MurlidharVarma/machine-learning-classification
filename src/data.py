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


def make_split(encoded, deduplicate=True, test_size=TEST_SIZE, seed=SEED):
    """Stratified train/test split, returning (X_train, X_test, y_train, y_test).

    deduplicate=True is the default and the headline configuration. 269 of the
    520 records are exact repeats of another record, so a plain split places
    identical rows on both sides and the resulting test score partly measures
    memorisation rather than generalisation. Dropping duplicates before the split
    guarantees no response profile appears in both halves.

    deduplicate=False reproduces the conventional split for the contrast reported
    in the README.

    Stratified because the classes are uneven and the dataset is small enough
    that an unstratified draw shifts the test-set balance noticeably.
    """
    frame = encoded.drop_duplicates() if deduplicate else encoded

    return train_test_split(
        frame[FEATURES],
        frame[TARGET],
        test_size=test_size,
        stratify=frame[TARGET],
        random_state=seed,
    )


def describe_problem(frame, require_target=True):
    """Return a readable reason the frame cannot be scored, or None if it is fine.

    Lives here rather than in app.py so the checks are testable without starting
    a Streamlit session, and so the app and the training code agree on what a
    valid input file looks like.
    """
    if frame.empty:
        return "The file contains no rows."

    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        return f"Missing {len(missing)} required feature column(s): " + ", ".join(missing)

    if require_target and TARGET not in frame.columns:
        return (
            f"No '{TARGET}' column found. Upload the labelled test split — metrics "
            "cannot be computed without the true labels."
        )

    return None


def unmapped_columns(encoded):
    """Columns holding NaN after encoding, i.e. those with unexpected values."""
    checked = [column for column in FEATURES + [TARGET] if column in encoded.columns]

    return [column for column in checked if encoded[column].isna().any()]
