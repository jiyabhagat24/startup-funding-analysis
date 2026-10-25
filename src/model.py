"""Random Forest model: will a funding round be a HIGH-funding round?

Target
    high_funding = 1 if the round's amount is above the median round size of the
    training set (about US$1.8M), else 0.

Features (things you would know before the round closes)
    funding_stage, sector, country, region, revenue_status,
    female_founder, employee_count, age_at_round, year

Deliberately NOT used (they are outcomes of the deal, so they would leak the answer)
    pre/post-money valuation, equity dilution, number of investors, lead investor,
    investor type.

Validation
    Rounds from the same startup never appear in both train and test
    (GroupShuffleSplit / GroupKFold on startup_id).

Run from the project root:  python src/model.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import BLUE, FIG_DIR, INK_2, METRICS_PATH, MODEL_DIR, MODEL_PATH, ORANGE, apply_style  # noqa: E402
from src.data_prep import load_master  # noqa: E402

CAT_FEATURES = ["funding_stage", "sector", "country", "region", "revenue_status"]
NUM_FEATURES = ["female_founder", "employee_count", "age_at_round", "year"]
FEATURES = CAT_FEATURES + NUM_FEATURES
RANDOM_STATE = 42


def make_xy(df, threshold):
    X = df[FEATURES].copy()
    X["funding_stage"] = X["funding_stage"].astype(str)
    y = (df["amount_million_usd"] > threshold).astype(int)
    return X, y


def make_pipeline(features=None, **rf_params):
    features = features or FEATURES
    cats = [c for c in features if c in CAT_FEATURES]
    nums = [c for c in features if c in NUM_FEATURES]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats)] + ([("num", "passthrough", nums)] if nums else []))
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **rf_params)
    return Pipeline([("prep", pre), ("rf", rf)])


def metrics_for(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return dict(accuracy=round(accuracy_score(y_true, pred), 4), precision=round(precision_score(y_true, pred), 4),
                recall=round(recall_score(y_true, pred), 4), f1=round(f1_score(y_true, pred), 4),
                roc_auc=round(roc_auc_score(y_true, proba), 4))


def train(df=None, save=True, make_figures=True):
    df = load_master() if df is None else df
    groups = df["startup_id"]
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE).split(df, groups=groups))
    tr, te = df.iloc[train_idx], df.iloc[test_idx]
    threshold = float(tr["amount_million_usd"].median())
    Xtr, ytr = make_xy(tr, threshold)
    Xte, yte = make_xy(te, threshold)
    assert set(tr["startup_id"]).isdisjoint(set(te["startup_id"]))

    grid = {"rf__n_estimators": [200, 400], "rf__max_depth": [None, 8, 12], "rf__min_samples_leaf": [3, 5, 10]}
    search = GridSearchCV(make_pipeline(), grid, scoring="roc_auc", cv=GroupKFold(n_splits=5), n_jobs=-1)
    search.fit(Xtr, ytr, groups=tr["startup_id"])
    model = search.best_estimator_

    proba = model.predict_proba(Xte)[:, 1]
    test_metrics = metrics_for(yte, proba)
    cm = confusion_matrix(yte, (proba >= 0.5).astype(int))

    # Baselines, so the score can be judged honestly.
    majority = float(max(yte.mean(), 1 - yte.mean()))
    stage_only = make_pipeline(["funding_stage"], n_estimators=200, min_samples_leaf=5)
    stage_only.fit(Xtr[["funding_stage"]], ytr)
    stage_proba = stage_only.predict_proba(Xte[["funding_stage"]])[:, 1]
    stage_metrics = metrics_for(yte, stage_proba)

    # Harder question: is a round large *for its own stage*? (target = above the stage median)
    stage_med = tr.groupby("funding_stage", observed=True)["amount_million_usd"].median()
    rel_tr = (tr["amount_million_usd"] > tr["funding_stage"].map(stage_med).astype(float)).astype(int)
    rel_te = (te["amount_million_usd"] > te["funding_stage"].map(stage_med).astype(float)).astype(int)
    rel_model = make_pipeline(**{k.replace("rf__", ""): v for k, v in search.best_params_.items()})
    rel_model.fit(Xtr, rel_tr)
    rel_auc = float(roc_auc_score(rel_te, rel_model.predict_proba(Xte)[:, 1]))

    # Permutation importance on held-out data (drop in AUC when a column is shuffled).
    imp = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1)
    importance = (pd.DataFrame({"feature": FEATURES, "auc_drop": imp.importances_mean, "std": imp.importances_std})
                  .sort_values("auc_drop", ascending=False).reset_index(drop=True))

    info = dict(
        target=f"high_funding = round amount > US${threshold:.2f}M (median of training rounds)",
        threshold_million_usd=round(threshold, 4),
        features=FEATURES, cat_features=CAT_FEATURES, num_features=NUM_FEATURES,
        categories={c: sorted(df[c].astype(str).unique().tolist()) for c in CAT_FEATURES},
        n_total=int(len(df)), n_train=int(len(tr)), n_test=int(len(te)),
        n_startups_train=int(tr["startup_id"].nunique()), n_startups_test=int(te["startup_id"].nunique()),
        positive_rate_train=round(float(ytr.mean()), 4), positive_rate_test=round(float(yte.mean()), 4),
        best_params={k.replace("rf__", ""): v for k, v in search.best_params_.items()},
        cv_roc_auc_best=round(float(search.best_score_), 4),
        test=test_metrics, confusion_matrix=cm.tolist(),
        baselines=dict(majority_class_accuracy=round(majority, 4), stage_only_random_forest=stage_metrics),
        stage_relative_experiment=dict(
            description="Same features, but target = round is above the median for its own funding stage",
            test_roc_auc=round(rel_auc, 4)),
        permutation_importance=importance.round(4).to_dict("records"),
        sklearn_version=sklearn.__version__,
    )
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(info, indent=2))
    if make_figures:
        _figures(model, Xte, yte, proba, cm, importance, stage_proba, stage_metrics, test_metrics)
    return model, info


def _figures(model, Xte, yte, proba, cm, importance, stage_proba, stage_metrics, test_metrics):
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 5))
    fpr, tpr, _ = roc_curve(yte, proba)
    fpr2, tpr2, _ = roc_curve(yte, stage_proba)
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"Full model (AUC {test_metrics['roc_auc']:.3f})")
    ax.plot(fpr2, tpr2, color=ORANGE, lw=2, ls="--", label=f"Funding stage only (AUC {stage_metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="#9a9893", lw=1)
    ax.set(title="ROC curve on held-out startups", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG_DIR / "model_roc_curve.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    labels = ["Lower", "Higher"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    ax.set(title="Confusion matrix (test set)", xlabel="Predicted round size", ylabel="Actual round size")
    fig.savefig(FIG_DIR / "model_confusion_matrix.png"); plt.close(fig)

    imp = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(imp["feature"], imp["auc_drop"], xerr=imp["std"], color=[ORANGE if f == "funding_stage" else BLUE for f in imp["feature"]],
            error_kw=dict(ecolor="#9a9893", lw=1))
    for y, v in enumerate(imp["auc_drop"]):
        ax.text(max(v, 0) + 0.005, y, f"{v:.3f}", va="center", fontsize=8.5, color=INK_2)
    ax.set(title="Permutation importance: drop in test AUC when shuffled", xlabel="Decrease in ROC AUC", ylabel="")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG_DIR / "model_feature_importance.png"); plt.close(fig)


def load_or_train():
    """Used by the dashboard. Loads the saved model; retrains if it is missing or was
    saved with a different scikit-learn version (pickles are version-specific)."""
    try:
        info = json.loads(METRICS_PATH.read_text())
        if info.get("sklearn_version") != sklearn.__version__:
            raise RuntimeError("scikit-learn version changed")
        return joblib.load(MODEL_PATH), info
    except Exception:
        return train(save=True, make_figures=False)


def predict_proba_high(model, **row):
    X = pd.DataFrame([row])[FEATURES]
    return float(model.predict_proba(X)[0, 1])


if __name__ == "__main__":
    _, info = train()
    print(json.dumps({k: v for k, v in info.items() if k not in ("categories",)}, indent=2))
