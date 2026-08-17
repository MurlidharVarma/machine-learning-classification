"""Streamlit front end for the early-stage diabetes risk classifiers.

Loads the five pipelines fitted by model/train.py and scores them against an
uploaded test file. Training does not normally happen here: the app would
otherwise report different numbers from the ones in the README.
"""

import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from src.config import (
    FEATURES,
    GENDER_FEATURE,
    NEGATIVE_LABEL,
    NUMERIC_FEATURES,
    POSITIVE_LABEL,
    STUDENT_ID,
    STUDENT_NAME,
    SYMPTOM_FEATURES,
    TARGET,
    TEST_DATA_PATH,
)
from src.data import (
    describe_problem,
    encode_features,
    load_data,
    make_split,
    unmapped_columns,
)
from src.metrics import METRIC_ORDER, compute_metrics, evaluate_all, metrics_table
from src.models import build_pipelines, model_path
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

ACCENT = "#D85A30"
ACCENT_DARK = "#993C1D"
ACCENT_DEEP = "#4A1B0C"
ACCENT_PALE = "#FAECE7"

st.set_page_config(
    page_title="Early stage diabetes risk — Model comparison",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; max-width: 1180px; }}
      h1 {{ color: {ACCENT_DARK}; font-weight: 700; letter-spacing: -0.5px; font-size: 1.9rem; }}
      h2, h3 {{ color: #40566080; }}
      .lead {{
          border-left: 4px solid {ACCENT};
          padding: 0.3rem 0 0.3rem 0.9rem;
          color: #555; font-size: 0.93rem; margin-bottom: 1.2rem;
      }}
      div[data-testid="stMetricValue"] {{ font-size: 1.5rem; color: {ACCENT_DEEP}; }}
      div[data-testid="stMetric"] {{
          background: #FAFAF8; border-radius: 8px; padding: 0.6rem 0.8rem;
      }}
      .stTabs [data-baseweb="tab"] {{ font-size: 0.93rem; }}
      .stTabs [aria-selected="true"] {{ color: {ACCENT_DARK}; }}
      .idcard {{
          font-size: 0.78rem; color: #6b6b6b; line-height: 1.5;
          border-top: 1px solid #e4e4e4; padding-top: 0.7rem; margin-top: 0.5rem;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    """Load the five fitted pipelines once per server process.

    cache_resource rather than cache_data: these are live objects meant to be
    shared, not values to be copied per session.

    If any file is missing or refuses to unpickle -- the usual cause being a
    scikit-learn version on the serving side that differs from the one that
    wrote the file -- the pipelines are refitted from the bundled data instead.
    Fitting all five takes well under a tenth of a second on 175 rows, so the
    fallback costs nothing and prevents a version skew from taking the whole app
    down. The banner tells the user which path was taken, because refitted
    models are not guaranteed to be byte-identical to the committed ones.
    """
    try:
        models = {name: joblib.load(model_path(name)) for name in build_pipelines()}
        return models, None
    except Exception as error:  # noqa: BLE001 - any load failure takes the same route
        X_train, _, y_train, _ = make_split(encode_features(load_data()))
        models = build_pipelines()
        for pipeline in models.values():
            pipeline.fit(X_train, y_train)
        return models, str(error)


@st.cache_data
def read_csv(source):
    return pd.read_csv(source)


def confusion_figure(y_true, y_pred, title):
    figure, axis = plt.subplots(figsize=(3.1, 2.7))
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True, fmt="d", cbar=False, square=True,
        cmap=sns.light_palette(ACCENT, as_cmap=True),
        xticklabels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        yticklabels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        annot_kws={"size": 13}, linewidths=0.5, linecolor="white", ax=axis,
    )
    axis.set_xlabel("Predicted", fontsize=9)
    axis.set_ylabel("Actual", fontsize=9)
    axis.set_title(title, fontsize=10, color=ACCENT_DARK)
    axis.tick_params(labelsize=9)
    figure.tight_layout()
    return figure


def roc_figure(models, X, y, auc_by_model):
    figure, axis = plt.subplots(figsize=(4.6, 3.6))
    ordered = sorted(models, key=lambda n: auc_by_model[n], reverse=True)
    for rank, name in enumerate(ordered):
        false_positive, true_positive, _ = roc_curve(y, models[name].predict_proba(X)[:, 1])
        axis.plot(
            false_positive, true_positive,
            label=f"{name} ({auc_by_model[name]:.3f})",
            linewidth=2.0 if rank == 0 else 1.2,
            color=ACCENT if rank == 0 else None,
            zorder=3 if rank == 0 else 2,
        )
    axis.plot([0, 1], [0, 1], "--", color="#bbbbbb", linewidth=1, label="Chance (0.500)")
    axis.set_xlabel("False positive rate", fontsize=9)
    axis.set_ylabel("True positive rate", fontsize=9)
    axis.legend(fontsize=7.5, loc="lower right", frameon=False)
    axis.tick_params(labelsize=8)
    for edge in ("top", "right"):
        axis.spines[edge].set_visible(False)
    figure.tight_layout()
    return figure


models, fallback_reason = load_models()

st.title("Early stage diabetes risk — model comparison")
st.markdown(
    '<div class="lead">Five classifiers trained on the UCI Early Stage Diabetes Risk '
    "Prediction dataset. Duplicate response profiles were removed before splitting, so the "
    "scores below reflect generalisation rather than memorised rows.</div>",
    unsafe_allow_html=True,
)

if fallback_reason:
    st.warning(
        "Saved model files could not be loaded, so the pipelines were refitted from the "
        f"bundled training split. Metrics may differ marginally from the README. Cause: {fallback_reason}"
    )

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Test data (CSV)", type="csv")
    use_bundled = st.checkbox(
        "Use the bundled test_data.csv", value=False,
        help="Scores the 76-row held-out split shipped with the repository.",
    )
    st.caption(
        f"Expects the original column names and Yes/No values, including the `{TARGET}` "
        "column with the true labels."
    )

    st.divider()
    st.header("Model")
    selected = st.selectbox("Examine in detail", list(models))
    st.caption("All five are scored regardless of this choice.")

    st.markdown(
        f'<div class="idcard"><b>{STUDENT_NAME}</b><br>{STUDENT_ID}<br>'
        "Machine Learning — Assignment 2</div>",
        unsafe_allow_html=True,
    )

if uploaded is not None:
    raw = read_csv(uploaded)
    source_label = uploaded.name
elif use_bundled:
    raw = read_csv(TEST_DATA_PATH)
    source_label = TEST_DATA_PATH.name
else:
    raw = None

comparison_tab, detail_tab, assess_tab = st.tabs(
    ["Model comparison", "Selected model", "Single assessment"]
)

scores = {}

if raw is not None:
    problem = describe_problem(raw)
    if problem:
        st.error(problem)
        st.stop()

    encoded = encode_features(raw)
    unmapped = unmapped_columns(encoded)
    if unmapped:
        st.error(
            "Unrecognised values in: " + ", ".join(unmapped) +
            ". Symptom columns accept only 'Yes' or 'No'; Gender only 'Male' or 'Female'; "
            f"{TARGET} only '{POSITIVE_LABEL}' or '{NEGATIVE_LABEL}'."
        )
        st.stop()

    X, y = encoded[FEATURES], encoded[TARGET]
    scores = evaluate_all(models, X, y)
    baseline = max(y.mean(), 1 - y.mean())

with comparison_tab:
    if raw is None:
        st.info("Upload a CSV in the sidebar, or tick the bundled test file, to score the models.")
        with st.expander("What the file needs to contain"):
            st.write(
                f"The {len(FEATURES)} feature columns plus `{TARGET}`, in their original form — "
                f"`Yes`/`No` for the symptom indicators, `Male`/`Female` for Gender, and "
                f"`{POSITIVE_LABEL}`/`{NEGATIVE_LABEL}` for the target. The app applies the same "
                "encoding used during training."
            )
            st.table(load_data().head(5))
    else:
        st.success(
            f"Scored **{source_label}** — {len(raw)} records, "
            f"{int(y.sum())} {POSITIVE_LABEL}, {int((1 - y).sum())} {NEGATIVE_LABEL}."
        )

        table = metrics_table(scores)
        # st.table, not st.dataframe. The dataframe widget draws into a canvas
        # grid that intermittently fails to repaint after a tab switch, leaving
        # the row labels visible and every number blank. This table is the
        # headline result of the whole app, so it renders as plain HTML instead.
        st.table(
            table.style
            .format("{:.4f}")
            .background_gradient(cmap="Oranges", axis=0)
            .set_properties(**{"font-size": "0.92rem"})
        )

        best = table["MCC"].idxmax()
        st.markdown(
            f"Highest MCC on this file: **{best}** ({table.loc[best, 'MCC']:.4f}). "
            f"Majority-class baseline is {baseline:.1%} accuracy and 0.000 MCC — MCC is the "
            "primary comparison metric because it uses all four cells of the confusion matrix "
            "and cannot be inflated by the majority class alone."
        )

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**ROC curves — all five models**")
            st.pyplot(roc_figure(models, X, y, {n: s["AUC"] for n, s in scores.items()}))
        with right:
            st.markdown("**Ranking by each metric**")
            # Headers abbreviated so six numeric columns fit the half-width
            # container without the last one being clipped.
            short = {"Accuracy": "Acc", "Precision": "Prec", "Recall": "Rec"}
            st.table(
                pd.DataFrame({
                    short.get(metric, metric): table[metric].rank(ascending=False).astype(int)
                    for metric in METRIC_ORDER
                })
            )
            st.caption("1 = best. A model leading on every column is a robust winner, not a lucky split.")

with detail_tab:
    if raw is None:
        st.info("Upload a CSV in the sidebar, or tick the bundled test file, to score the models.")
    else:
        st.subheader(f"{selected}")

        probabilities = models[selected].predict_proba(X)[:, 1]

        threshold = st.slider(
            "Decision threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05,
            help="The probability above which a case is called Positive. Lowering it trades "
                 "false positives for fewer missed cases.",
        )
        thresholded = (probabilities >= threshold).astype(int)
        at_threshold = compute_metrics(y, thresholded, probabilities)

        # One row of tiles that tracks the slider, rather than a static row plus a
        # second identical row -- at the default 0.50 the two were the same numbers
        # twice. Deltas appear only once the slider has actually been moved.
        default = scores[selected]
        for column, metric in zip(st.columns(6), METRIC_ORDER):
            delta = at_threshold[metric] - default[metric]
            column.metric(
                metric, f"{at_threshold[metric]:.4f}",
                delta=None if abs(delta) < 5e-5 else f"{delta:+.4f}",
            )

        if abs(threshold - 0.50) < 1e-9:
            st.caption(
                f"Majority-class baseline on this file is {baseline:.1%} accuracy and 0.000 MCC. "
                "Accuracy is only meaningful above that floor."
            )
        else:
            st.caption(
                "Deltas are against the default 0.50 threshold. AUC is threshold-independent, "
                f"so it does not move. Baseline accuracy on this file is {baseline:.1%}."
            )

        st.divider()
        left, right = st.columns([1, 1.3])

        with left:
            st.markdown("**Confusion matrix**")
            st.pyplot(confusion_figure(y, thresholded, f"{selected} @ {threshold:.2f}"))

        with right:
            st.markdown("**Classification report**")
            report = classification_report(
                y, thresholded,
                target_names=[NEGATIVE_LABEL, POSITIVE_LABEL],
                output_dict=True, zero_division=0,
            )
            # The "accuracy" entry is a bare float among per-class dicts, so
            # DataFrame broadcasts it across all four columns and reports an
            # "accuracy precision" and an "accuracy support" that mean nothing.
            # Dropped here; accuracy already has its own tile above.
            report.pop("accuracy", None)
            report_frame = pd.DataFrame(report).T
            report_frame["support"] = report_frame["support"].astype(int)
            # st.table rather than st.dataframe: the latter paints into a canvas
            # grid that leaves its columns blank at this width in some browsers.
            # A four-row summary has nothing to gain from a scrollable grid.
            st.table(
                report_frame.style.format(
                    {"precision": "{:.3f}", "recall": "{:.3f}",
                     "f1-score": "{:.3f}", "support": "{:d}"}
                )
            )

            false_negatives = int(((y == 1) & (thresholded == 0)).sum())
            false_positives = int(((y == 0) & (thresholded == 1)).sum())
            st.caption(
                f"{false_negatives} missed case(s) and {false_positives} unnecessary referral(s) "
                "at this threshold. A missed case receives no follow-up; a false positive costs "
                "one blood test."
            )

with assess_tab:
    st.subheader("Score a single questionnaire")
    st.caption(
        "Independent of the uploaded file — this runs the selected model against values you "
        "enter directly."
    )

    left, right = st.columns([1.4, 1])

    with left:
        age = st.slider("Age", min_value=16, max_value=90, value=48)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

        st.markdown("**Reported symptoms**")
        responses = {}
        symptom_columns = st.columns(2)
        for index, symptom in enumerate(SYMPTOM_FEATURES):
            with symptom_columns[index % 2]:
                responses[symptom] = st.checkbox(symptom, key=f"symptom_{index}")

    single = pd.DataFrame([{
        NUMERIC_FEATURES[0]: age,
        GENDER_FEATURE: gender,
        **{symptom: ("Yes" if flag else "No") for symptom, flag in responses.items()},
    }])[FEATURES]

    encoded_single = encode_features(single)
    probability = float(models[selected].predict_proba(encoded_single)[:, 1][0])
    verdict = POSITIVE_LABEL if probability >= 0.5 else NEGATIVE_LABEL

    with right:
        st.markdown(
            f"""
            <div style="background:{ACCENT_PALE};border-radius:12px;padding:1.1rem;text-align:center">
              <div style="font-size:0.8rem;color:{ACCENT_DARK}">Predicted risk — {selected}</div>
              <div style="font-size:2.4rem;font-weight:600;color:{ACCENT_DEEP};line-height:1.3">{probability:.2f}</div>
              <div style="font-size:0.85rem;color:{ACCENT_DARK};margin-bottom:0.6rem">{verdict}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(probability)
        st.caption(
            "A screening aid, not a diagnosis. The dataset comes from a single hospital in "
            "Sylhet, Bangladesh, and its cohort is not representative of the general population."
        )
