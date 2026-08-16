"""
Train, evaluate and persist the five classifiers.

Writes five .joblib pipelines and metadata.json into model/, and the held-out
test split to test_data.csv at the project root.
"""

import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

# To avoid hardcoding of src path and survives the notebook being moved.
_here = Path(__file__).resolve()
for _candidate in [_here.parent, *_here.parents]:
    if (_candidate / "src").is_dir():
        sys.path.insert(0, str(_candidate))
        break

import joblib  # noqa: E402
import sklearn  # noqa: E402

from src.config import (  # noqa: E402
    DATA_PATH,
    FEATURES,
    METADATA_PATH,
    MODEL_DIR,
    SEED,
    STUDENT_ID,
    STUDENT_NAME,
    TARGET,
    TEST_DATA_PATH,
    TEST_SIZE,
)
from src.data import encode_features, load_data, make_split  # noqa: E402
from src.metrics import METRIC_ORDER, evaluate_all, metrics_table  # noqa: E402
from src.models import build_pipelines, model_path  # noqa: E402

def banner():
    line = "=" * 78
    print(line)
    print("  Early Stage Diabetes Risk Prediction - Machine Learning Assignment 2")
    print(f"  {STUDENT_NAME}  |  {STUDENT_ID}")
    print(f"  Run started {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Python {sys.version.split()[0]}  |  scikit-learn {sklearn.__version__}")
    print(line)


def section(number, title):
    print(f"\n{number}. {title}\n{'-' * 78}")


def describe_dataset(raw):
    section(1, "Dataset")

    duplicated = int(raw.duplicated().sum())
    counts = raw[TARGET].value_counts()

    print(f"Source          {DATA_PATH.name}")
    print(f"Shape           {raw.shape[0]} instances x {raw.shape[1] - 1} features + target")
    print(f"Missing values  {int(raw.isna().sum().sum())}")
    print(f"Class balance   {counts.to_dict()}  majority {counts.max() / len(raw):.4f}")
    print(f"Duplicate rows  {duplicated} exact repeats, leaving {len(raw) - duplicated} distinct profiles")

    # If any repeated profile carried both labels the duplicates would be
    # contradictory labelling, which needs a different remedy from leakage.
    conflicting = int((raw.groupby(FEATURES, observed=True)[TARGET].nunique() > 1).sum())
    print(f"Conflicts       {conflicting} profiles carry more than one label")


def build_splits(encoded):
    section(2, "Split")

    X_train, X_test, y_train, y_test = make_split(encoded, deduplicate=True)

    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"Deduplicated first, then stratified {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}, seed {SEED}.")
    print(f"Train           {len(X_train)} rows ({int(y_train.sum())} Positive)")
    print(f"Test            {len(X_test)} rows ({int(y_test.sum())} Positive)")
    print(f"Baseline        {baseline:.4f} accuracy for always predicting the majority class (MCC 0.000)")

    return X_train, X_test, y_train, y_test


def train(X_train, y_train):
    section(3, "Training")

    pipelines = build_pipelines()
    started = time.perf_counter()
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        print(f"  fitted  {name}")
    print(f"\nAll five fitted in {time.perf_counter() - started:.3f}s.")

    return pipelines


def evaluate(pipelines, X_test, y_test):
    section(4, "Comparison table - held-out test set")

    table = metrics_table(evaluate_all(pipelines, X_test, y_test))
    print(table.round(4).to_string())
    print(f"\nBest MCC: {table['MCC'].idxmax()} ({table['MCC'].max():.4f})")

    return table


def persist(pipelines, X_train, X_test, raw, table):
    section(5, "Persistence")

    MODEL_DIR.mkdir(exist_ok=True)
    for name, pipeline in pipelines.items():
        destination = model_path(name)
        joblib.dump(pipeline, destination)
        print(f"  wrote  model/{destination.name}")

    test_rows = raw.loc[X_test.index]
    test_rows.to_csv(TEST_DATA_PATH, index=False)
    print(f"  wrote  {TEST_DATA_PATH.name} ({len(test_rows)} rows, labelled, original form)")

    metadata = {
        "student_name": STUDENT_NAME,
        "student_id": STUDENT_ID,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "deduplicated": True,
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": {name: {k: round(v, 6) for k, v in row.items()} for name, row in table.iterrows()},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote  model/{METADATA_PATH.name}")


def main():
    banner()

    raw = load_data()
    describe_dataset(raw)

    encoded = encode_features(raw)
    X_train, X_test, y_train, y_test = build_splits(encoded)

    pipelines = train(X_train, y_train)
    table = evaluate(pipelines, X_test, y_test)
    persist(pipelines, X_train, X_test, raw, table)

    print(f"\n{'=' * 78}")
    print(f"  Complete. Metrics reported: {', '.join(METRIC_ORDER)}.")
    print(f"  {STUDENT_NAME}  |  {STUDENT_ID}")
    print("=" * 78)


if __name__ == "__main__":
    main()
