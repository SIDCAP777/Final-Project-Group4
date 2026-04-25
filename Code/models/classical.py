# ============================================================
# classical.py
# Builders for classical baseline models.
#
# Each function returns an unfitted scikit-learn model with
# sensible defaults. Training/fitting happens in train_classical.py
#
# Models:
#   - Logistic Regression: linear, fast, interpretable, strong baseline
#   - Linear SVM: often the strongest classical text classifier
#   - Multinomial Naive Bayes: very fast, works well with TF-IDF/BoW
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV


def build_logistic_regression(seed: int = 42) -> LogisticRegression:
    # Logistic Regression with L2 regularization
    # - solver='liblinear' handles sparse TF-IDF efficiently
    # - C=1.0 is a reasonable default; we can tune later if time permits
    # - max_iter raised to ensure convergence on large feature sets
    return LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=1000,
        random_state=seed,
        n_jobs=1,  # liblinear is single-threaded
    )


def build_linear_svm(seed: int = 42) -> CalibratedClassifierCV:
    # LinearSVC has no predict_proba (it's a hinge-loss classifier),
    # but we need probabilities for LIME/SHAP later. Wrapping in
    # CalibratedClassifierCV gives us predict_proba via Platt scaling.
    base = LinearSVC(
        C=1.0,
        max_iter=2000,
        random_state=seed,
    )
    # cv=3 keeps calibration training cheap on large datasets
    return CalibratedClassifierCV(base, cv=3)


def build_naive_bayes() -> MultinomialNB:
    # Multinomial NB: assumes features are non-negative counts/frequencies.
    # TF-IDF values are non-negative, so this works (though traditionally NB
    # is paired with raw counts -- still a strong baseline either way).
    return MultinomialNB(alpha=1.0)  # alpha=1.0 is Laplace smoothing


def get_model(name: str, seed: int = 42):
    # Factory function: pick a model by name string.
    # Used by the training script so we can loop over a list of names.
    name = name.lower()
    if name == "logistic_regression":
        return build_logistic_regression(seed=seed)
    elif name == "linear_svm":
        return build_linear_svm(seed=seed)
    elif name == "naive_bayes":
        return build_naive_bayes()
    else:
        raise ValueError(f"Unknown classical model: {name}")
