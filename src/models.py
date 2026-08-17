"""The five required classifiers, each wrapped in a Pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import MODEL_DIR, NUMERIC_FEATURES, SEED

def _scale_age():
    return ColumnTransformer(
        [("scale_age", StandardScaler(), NUMERIC_FEATURES)],
        remainder="passthrough",
    )


def build_pipelines():
    """Return {display name: unfitted Pipeline} for all five required models.

    Wrapping every model in a Pipeline is a correctness requirement, not tidiness.
    Inside a Pipeline the scaler is refitted on the training portion
    of every cross-validation fold automatically.

    Scaling is applied only where the algorithm is sensitive to feature scale
    """
    return {
        # Logistic Regression - the L2 penalty shrinks coefficients by magnitude, so it penalises features evenly only on a shared scale.
        # liblinear is the documented choice for small datasets and produces none of numpy overflow warning issues with matrix matmuls, 
        # which keeps the training run readable.
        "Logistic Regression": Pipeline([
            ("preprocess", _scale_age()),
            ("classifier", LogisticRegression(
                solver="liblinear", max_iter=1000, random_state=SEED
            )),
        ]),
        # Decision Tree - split on a threshold within a single feature, so the
        # relative scale between features is irrelevant
        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(random_state=SEED)),
        ]),
        # kNN - Euclidean distance is dominated by the widest-ranging feature. 
        # Unscaled, Age (16-90) would swamp fifteen 0/1 columns 
        # and Age separates the classes least well of any feature, 
        # so neighbours would be ranked on noise.
        "kNN": Pipeline([
            ("preprocess", _scale_age()),
            ("classifier", KNeighborsClassifier()),
        ]),
        # Gaussian NB - fits a mean and variance per feature per class; a linear rescaling shifts both and changes nothing.
          "Naive Bayes": Pipeline([
            ("classifier", GaussianNB()),
        ]),
        "Random Forest": Pipeline([
            ("classifier", RandomForestClassifier(random_state=SEED)),
        ]),
    }

def model_path(name):
    """Map a display name to its .joblib path.

    Derived rather than held in a lookup table, so the script that writes the
    files and the app that reads them cannot disagree about a filename.
    """
    return MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib"