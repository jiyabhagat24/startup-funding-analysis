"""Compare Random Forest with other classifiers on the SAME data, split and target.

Every model gets the same train/test split (grouped by startup), the same features,
and a small grid search scored with 5-fold GroupKFold on the training data only.

Run from the project root:  python src/compare_models.py
Outputs: reports/model_comparison.json, figures/model_comparison.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import BLUE, FIG_DIR, INK_2, ORANGE, REPORT_DIR, apply_style  # noqa: E402
from src.data_prep import load_master  # noqa: E402
from src.model import CAT_FEATURES, NUM_FEATURES, RANDOM_STATE, make_xy, metrics_for  # noqa: E402


def preprocessor(cats, nums, scale):
    parts = [("cat", OneHotEncoder(handle_unknown="ignore"), cats)]
    if nums:
        parts.append(("num", StandardScaler() if scale else "passthrough", nums))
    return ColumnTransformer(parts)


def main():
    df = load_master()
    tr_i, te_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE).split(df, groups=df["startup_id"]))
    tr, te = df.iloc[tr_i], df.iloc[te_i]
    thr = float(tr["amount_million_usd"].median())
    Xtr, ytr = make_xy(tr, thr)
    Xte, yte = make_xy(te, thr)

    specs = {
        "Logistic Regression": (LogisticRegression(max_iter=3000), {"clf__C": [0.1, 1, 10]}, True, CAT_FEATURES, NUM_FEATURES),
        "Logistic Regression (stage only)": (LogisticRegression(max_iter=3000), {"clf__C": [0.1, 1, 10]}, True, ["funding_stage"], []),
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                          {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, False, CAT_FEATURES, NUM_FEATURES),
        "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                          {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 8, 12], "clf__min_samples_leaf": [3, 5, 10]},
                          False, CAT_FEATURES, NUM_FEATURES),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE),
                              {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [2, 3]},
                              False, CAT_FEATURES, NUM_FEATURES),
    }
    rows = []
    for name, (est, grid, scale, cats, nums) in specs.items():
        cols = cats + nums
        pipe = Pipeline([("prep", preprocessor(cats, nums, scale)), ("clf", est)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=GroupKFold(n_splits=5), n_jobs=-1)
        search.fit(Xtr[cols], ytr, groups=tr["startup_id"])
        m = metrics_for(yte, search.predict_proba(Xte[cols])[:, 1])
        rows.append(dict(model=name, cv_roc_auc=round(float(search.best_score_), 4), **{f"test_{k}": v for k, v in m.items()},
                         best_params={k.replace("clf__", ""): v for k, v in search.best_params_.items()}))
        print(f"{name:34s} CV AUC {search.best_score_:.3f} | test AUC {m['roc_auc']:.3f} | acc {m['accuracy']:.3f} | F1 {m['f1']:.3f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "model_comparison.json").write_text(json.dumps(rows, indent=2, default=str))

    apply_style()
    d = pd.DataFrame(rows).sort_values("test_roc_auc")
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.barh(d["model"], d["test_roc_auc"], color=[ORANGE if m == "Random Forest" else BLUE for m in d["model"]])
    for y, v in enumerate(d["test_roc_auc"]):
        ax.text(v, y, f"  {v:.3f}", va="center", fontsize=9, color=INK_2)
    ax.set(title="Test ROC AUC by model (same split, same target)", xlabel="ROC AUC (0.5 = guessing)", ylabel="")
    ax.set_xlim(0.5, 1.0)
    ax.grid(axis="y", visible=False)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "model_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
