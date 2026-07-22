"""
AI Trust Inspector — Student Performance audit dashboard.

Run with:  streamlit run dashboard.py

This dashboard audits the SYSTEM being evaluated (a pass/fail predictor
trained on the UCI Student Performance dataset) against performance,
fairness, and EU AI Act criteria. It is not a self-assessment of this
dashboard itself.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

from ucimlrepo import fetch_ucirepo
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairlearn.metrics import (
    MetricFrame, demographic_parity_difference, equalized_odds_difference,
)

st.set_page_config(page_title="AI Trust Inspector", layout="wide")


# ---------------------------------------------------------------------------
# Data + model (cached so the app doesn't reload/retrain on every widget click)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    student_performance = fetch_ucirepo(id=320)
    X_raw = student_performance.data.features
    y_raw = student_performance.data.targets

    G3 = y_raw["G3"] if "G3" in y_raw.columns else X_raw["G3"]
    y = (G3 >= 10).astype(int)

    leak_cols = [c for c in ["G1", "G2", "G3"] if c in X_raw.columns]
    features = X_raw.drop(columns=leak_cols)

    if "sex" not in features.columns:
        raise ValueError(
            "Expected a 'sex' column in the features — check the dataset "
            "schema with features.columns.tolist() if this fires."
        )

    return features, y


@st.cache_resource
def train_model(features, y, sensitive_col="sex"):
    sensitive_feature = features[sensitive_col]

    X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
        features, y, sensitive_feature,
        test_size=0.3, random_state=42, stratify=y,
    )

    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ])

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return clf, X_train, X_test, y_train, y_test, sf_train, sf_test, y_pred


@st.cache_data
def compute_cv_fail_recall(_clf, features, y, sensitive_col="sex", n_splits=5):
    """Cross-validated fail-class recall by group.

    Clones the pipeline fresh for every fold rather than reusing the fitted
    `_clf` — reusing it would leave the model trained on whichever fold ran
    last, which would silently break the SHAP explanation computed later
    from the original 70/30 split.
    """
    sensitive_feature = features[sensitive_col]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {group: [] for group in sensitive_feature.unique()}

    for train_idx, test_idx in skf.split(features, y):
        X_tr, X_te = features.iloc[train_idx], features.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        sf_te = sensitive_feature.iloc[test_idx]

        fold_clf = clone(_clf)
        fold_clf.fit(X_tr, y_tr)
        y_pr = fold_clf.predict(X_te)

        for group in results:
            mask = (sf_te == group).values
            if mask.sum() > 0:
                r = recall_score(y_te[mask], y_pr[mask], pos_label=0, zero_division=0)
                results[group].append(r)

    return {group: float(np.mean(vals)) for group, vals in results.items()}


# ---------------------------------------------------------------------------
# Regulatory findings — an editorial layer, not a live computation.
# This is where the numbers above get translated into something a human
# reader (and the EU AI Act) can make sense of. Update this list if you
# investigate further and find something new — everything below reads
# from this list, so a 4th entry just becomes a 4th card automatically.
# ---------------------------------------------------------------------------

REGULATORY_FINDINGS = [
    {
        "finding": "Fail-class recall is only ~27%, while headline accuracy "
                   "(81%) and F1 (0.89) look strong because they're computed "
                   "mostly against the majority 'pass' class.",
        "severity": "High",
        "eu_ai_act_reference": "Article 15 — accuracy must be appropriate to "
                                "the system's intended purpose, not just high "
                                "on average.",
        "interpretation": "If this system exists to flag at-risk students, its "
                           "real-world accuracy on that task is ~27%, not 81%.",
    },
    {
        "finding": "Across 5-fold cross-validation, fail-class recall for male "
                   "students exceeded female students in every fold (mean "
                   "0.375 vs 0.168), despite female students having higher "
                   "overall accuracy.",
        "severity": "High",
        "eu_ai_act_reference": "Recital 56 and Article 15 — high-risk "
                                "education systems must not reproduce "
                                "discriminatory patterns, and accuracy must be "
                                "assessed against the system's actual purpose.",
        "interpretation": "The direction of this gap held in all 5 folds — "
                           "it isn't an artifact of one lucky split. Overall "
                           "accuracy actively concealed it: the group with "
                           "better accuracy is the one the model fails hardest "
                           "at protecting.",
    },
    {
        "finding": "SHAP ranks 'school' as the top predictor — above the "
                   "student's own prior failures — and mother's education "
                   "(Medu) both ranks in the top 6 and has its own "
                   "Demographic Parity Difference of 0.215.",
        "severity": "High",
        "eu_ai_act_reference": "Article 10(2) — providers must examine "
                                "training data for proxies of protected or "
                                "socioeconomic characteristics.",
        "interpretation": "School and parental education plausibly function "
                           "as socioeconomic-status proxies, and the model "
                           "leans on them more than the student's own record "
                           "— a plausible mechanism behind the fairness gap "
                           "above.",
    },
]

SEVERITY_ICON = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}


# ---------------------------------------------------------------------------
# Load + train
# ---------------------------------------------------------------------------

features, y = load_data()
clf, X_train, X_test, y_train, y_test, sf_train, sf_test, y_pred = train_model(features, y)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
fail_recall = recall_score(y_test, y_pred, pos_label=0)
cm = confusion_matrix(y_test, y_pred)

dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sf_test)
eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sf_test)
mf = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=y_pred, sensitive_features=sf_test)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("AI Trust Inspector")
st.sidebar.markdown("**Dataset:** UCI Student Performance (id 320)")
st.sidebar.markdown("**Model:** Logistic Regression")
st.sidebar.markdown("**Protected attribute:** sex")
st.sidebar.markdown("**Target:** pass/fail (G3 ≥ 10), predicted *without* G1/G2")
st.sidebar.caption(
    "G1/G2 (earlier-period grades) are dropped on purpose — leaving them in "
    "would let the model just echo prior grades back, hiding any real "
    "demographic effect and misrepresenting the realistic early-warning "
    "use case this system stands in for."
)


# ---------------------------------------------------------------------------
# Header + Audit Summary
# ---------------------------------------------------------------------------

st.title("AI Trust Inspector")
st.caption("Can this model be trusted in production? Accuracy alone doesn't answer that.")

st.header("Audit Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall Accuracy", f"{acc:.1%}")
c2.metric(
    "Fail-Class Recall", f"{fail_recall:.1%}",
    help="The metric that actually matters if this system exists to catch at-risk students.",
)
c3.metric("Equalized Odds Diff. (sex)", f"{eod:.3f}")
c4.metric("Explainability", "Available (SHAP)")

st.warning(
    "🔴 **Recommendation: do not deploy as-is.** This model misses most "
    "at-risk students overall, and misses them substantially more often "
    "among female students specifically — a gap that held up under "
    "cross-validation. See the sections below for the full evidence."
)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

st.header("Performance")
p1, p2, p3, p4 = st.columns(4)
p1.metric("Accuracy", f"{acc:.1%}")
p2.metric("Precision (pass)", f"{prec:.1%}")
p3.metric("Recall (pass)", f"{rec:.1%}")
p4.metric("F1 (pass)", f"{f1:.2f}")

st.markdown(
    f"⚠️ These four numbers are all computed against the majority **pass** "
    f"class (~85% of students). Fail-class recall — the ability to catch "
    f"students who actually fail — is only **{fail_recall:.1%}**."
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual: Fail", "Actual: Pass"],
    columns=["Predicted: Fail", "Predicted: Pass"],
)
st.dataframe(cm_df)


# ---------------------------------------------------------------------------
# Fairness
# ---------------------------------------------------------------------------

st.header("Fairness")
fc1, fc2 = st.columns(2)
fc1.metric("Demographic Parity Difference (sex)", f"{dpd:.3f}")
fc2.metric("Equalized Odds Difference (sex)", f"{eod:.3f}")

st.markdown("**Accuracy by group** — looks close. This is the trap:")
st.dataframe(mf.by_group.rename("accuracy").to_frame().style.format("{:.1%}"))

with st.spinner("Running 5-fold cross-validation for fail-class recall by group..."):
    cv_results = compute_cv_fail_recall(clf, features, y)

st.markdown(
    "**Fail-class recall by group, cross-validated (5 folds)** — this is "
    "where the real gap shows up, once you condition on students who "
    "actually failed:"
)
cv_df = pd.DataFrame.from_dict(cv_results, orient="index", columns=["mean fail-class recall"])
st.dataframe(cv_df.style.format("{:.1%}"))


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

st.header("Explainability")

preprocessor_fitted = clf.named_steps["preprocessor"]
model_fitted = clf.named_steps["classifier"]
feature_names = preprocessor_fitted.get_feature_names_out()
X_train_transformed = preprocessor_fitted.transform(X_train)
X_test_transformed = preprocessor_fitted.transform(X_test)

explainer = shap.LinearExplainer(model_fitted, X_train_transformed)
shap_values = explainer(X_test_transformed)

plt.figure()
shap.summary_plot(
    shap_values, X_test_transformed, feature_names=feature_names,
    plot_type="bar", show=False,
)
st.pyplot(plt.gcf())
plt.close()

st.caption(
    "Note where 'school' and 'Medu' (mother's education) rank relative to "
    "the student's own history — that's the lead investigated in the EU AI "
    "Act Lens below."
)


# ---------------------------------------------------------------------------
# EU AI Act Lens
# ---------------------------------------------------------------------------

st.header("EU AI Act Lens")
st.caption(
    "This section maps findings above to specific EU AI Act provisions. "
    "It is an educational mapping exercise, not a legal compliance "
    "assessment — the author is not a lawyer, and this is not a "
    "conformity assessment."
)

for item in REGULATORY_FINDINGS:
    with st.container(border=True):
        st.markdown(f"{SEVERITY_ICON[item['severity']]} **{item['finding']}**")
        st.caption(f"📖 {item['eu_ai_act_reference']}")
        st.write(item["interpretation"])
