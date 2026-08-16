"""
The 6 evaluation metrics used to score the five required classifiers
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Positive is encoded as 1 by data.CLASS_MAP. Pinned explicitly so that
# precision, recall and F1 always describe the clinically relevant class rather
# than whichever label scikit-learn happens to sort first.
POSITIVE_CLASS = 1

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

"""
    Return the six metrics as a dict.

    y_score holds predicted probabilities for the positive class, not labels.
    AUC measures how well a model ranks cases across every possible threshold,
    so passing hard 0/1 predictions collapses it to a much less informative
    quantity that happens not to raise an error.

    MCC is the key metric. The majority class is roughly 62% of
    the full dataset and 69% after deduplication, so a model that predicts
    Positive unconditionally scores that as accuracy while scoring 0.000 MCC.
    MCC draws on all four cells of the confusion matrix and has no such blind
    spot.
"""
def compute_metrics(y_true, y_pred, y_score):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS),
        "Recall": recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS),
        "F1": f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

"""Score one fitted pipeline on (X, y)."""
def evaluate_pipeline(pipeline, X, y):
    return compute_metrics(y, pipeline.predict(X), pipeline.predict_proba(X)[:, 1])


"""Score every fitted pipeline in {name: pipeline} against the same data."""
def evaluate_all(pipelines, X, y):
    return {name: evaluate_pipeline(pipeline, X, y) for name, pipeline in pipelines.items()}


"""Turn {model name: metrics dict} into a DataFrame with metrics as columns."""
def metrics_table(results):
    return pd.DataFrame(results).T[METRIC_ORDER]
