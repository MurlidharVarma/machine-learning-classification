"""
All configs at one place:
Paths, column names and the constants that must not drift between modules.
"""

from pathlib import Path

STUDENT_NAME = "Murlidhar Ravi Geetha Varma"
STUDENT_ID = "2025AC05598"

REPO_URL = "https://github.com/MurlidharVarma/machine-learning-classification"
TEST_DATA_ON_GITHUB = f"{REPO_URL}/blob/main/test_data.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "diabetes_data_upload.csv"
MODEL_DIR = PROJECT_ROOT / "model"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
METADATA_PATH = MODEL_DIR / "metadata.json"

# One seed for all random operations, to ensure reproducibility.
SEED = 5598

# 0.3 rather than the more usual 0.2: deduplication leaves 251 distinct rows, and
# a 20% test set of 50 records makes each single record worth 2 points of accuracy
TEST_SIZE = 0.3

TARGET = "class"
POSITIVE_LABEL = "Positive"
NEGATIVE_LABEL = "Negative"

# Column names exactly as they appear in the UCI header. Casing is inconsistent
# import these rather than retyping them would be easier
NUMERIC_FEATURES = ["Age"]

GENDER_FEATURE = "Gender"

SYMPTOM_FEATURES = [
    "Polyuria",
    "Polydipsia",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity",
]

FEATURES = NUMERIC_FEATURES + [GENDER_FEATURE] + SYMPTOM_FEATURES

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest",
]
