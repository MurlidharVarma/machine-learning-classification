"""
All configs at one place:
Paths, column names and the constants that must not drift between modules.
"""

from pathlib import Path

STUDENT_NAME = "Murlidhar Ravi Geetha Varma"
STUDENT_ID = "2025AC05598"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "diabetes_data_upload.csv"
MODEL_DIR = PROJECT_ROOT / "model"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"

# One seed for all random operations, to ensure reproducibility
SEED = 5598

TEST_SIZE = 0.2

TARGET = "class"
POSITIVE_LABEL = "Positive"
NEGATIVE_LABEL = "Negative"

# Column names exactly as they appear in the UCI header. Casing is inconsistent
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
