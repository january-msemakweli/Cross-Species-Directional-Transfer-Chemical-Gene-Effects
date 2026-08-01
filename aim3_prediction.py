"""
Aim 3: can we predict, for a chemical-gene-aspect effect measured in rodents,
whether its direction will transfer to humans (independent evidence)?

Design that avoids leakage:
  - unit = effect key (both species, unambiguous, PMID-disjoint)
  - outcome = agree (human direction == rodent direction)
  - grouped CV by CHEMICAL (GroupKFold) so every test chemical is unseen
  - features known at prediction time: molecular aspect, chemical class,
    gene pathway pleiotropy, chemical/gene promiscuity, and rodent-side
    evidence depth. No human-side agreement information leaks in.
  - metrics: ROC-AUC, average precision, Brier, calibration; compared to a
    prevalence-only baseline. Plus leave-one-chemical-class-out generalization.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis_core import Config


def build_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    # promiscuity features from the FULL key table (pre-filter)
    chem_deg = df.groupby("chem_bare").size().rename("chem_promiscuity")
    gene_deg = df.groupby("gene").size().rename("gene_promiscuity")

    prim = df[df["in_human"] & df["in_rodent"] & df["pmid_disjoint_hr"]].copy()
    prim = prim[(prim["h_cons"] != 0) & (prim["rod_cons"] != 0)].copy()
    prim["agree"] = (prim["h_cons"] == prim["rod_cons"]).astype(int)

    prim = prim.merge(chem_deg, left_on="chem_bare", right_index=True, how="left")
    prim = prim.merge(gene_deg, left_on="gene", right_index=True, how="left")

    # rodent-side (predictor-side) evidence, plus generic depth
    prim["rod_support"] = prim["rod_inc"] + prim["rod_dec"]
    prim["rod_pmids_f"] = prim["rod_pmids"].astype(float)
    prim["log_chem_promiscuity"] = np.log1p(prim["chem_promiscuity"])
    prim["log_gene_promiscuity"] = np.log1p(prim["gene_promiscuity"])
    prim["log_pathway_deg"] = np.log1p(prim["gene_pathway_deg"])
    prim["rodent_direction"] = prim["rod_cons"].map({1: "increase", -1: "decrease"})
    return prim


NUM = [
    "rod_support",
    "rod_pmids_f",
    "log_chem_promiscuity",
    "log_gene_promiscuity",
    "log_pathway_deg",
]
CAT = ["aspect", "chem_class", "rodent_direction"]


def make_models():
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ]
    )
    logit = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0))])
    gb = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.08, max_iter=250,
        l2_regularization=1.0, early_stopping=True, n_iter_no_change=15,
        random_state=0,
    )
    # GB handles raw categoricals via ordinal wrapper: build a small pipeline
    from sklearn.preprocessing import OrdinalEncoder

    gb_pre = ColumnTransformer(
        [
            ("num", "passthrough", NUM),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
        ]
    )
    gb_pipe = Pipeline([("pre", gb_pre), ("clf", gb)])
    return {"logistic": logit, "gbm": gb_pipe}


def grouped_cv(prim: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    X = prim[NUM + CAT]
    y = prim["agree"].to_numpy()
    groups = prim["chem_bare"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    rows = []
    oof = {}
    for name, model in make_models().items():
        aucs, aps, briers = [], [], []
        oof_pred = np.full(len(y), np.nan)
        for tr, te in gkf.split(X, y, groups):
            model.fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[te])[:, 1]
            oof_pred[te] = p
            aucs.append(roc_auc_score(y[te], p))
            aps.append(average_precision_score(y[te], p))
            briers.append(brier_score_loss(y[te], p))
        oof[name] = oof_pred
        rows.append(
            {
                "model": name,
                "auc_mean": np.mean(aucs),
                "auc_sd": np.std(aucs),
                "ap_mean": np.mean(aps),
                "brier_mean": np.mean(briers),
            }
        )
    base_rate = y.mean()
    rows.append(
        {
            "model": "baseline_prevalence",
            "auc_mean": 0.5,
            "auc_sd": 0.0,
            "ap_mean": base_rate,
            "brier_mean": brier_score_loss(y, np.full_like(y, base_rate, dtype=float)),
        }
    )
    return pd.DataFrame(rows), oof


def calibration_table(y, p, bins=10):
    q = pd.qcut(pd.Series(p), bins, duplicates="drop")
    d = pd.DataFrame({"y": y, "p": p, "bin": q})
    g = d.groupby("bin", observed=True).agg(
        n=("y", "size"), pred=("p", "mean"), obs=("y", "mean")
    )
    return g.reset_index(drop=True)


def loco_generalization(prim: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-chemical-class-out: train on all classes but one, predict it."""
    X = prim[NUM + CAT]
    y = prim["agree"].to_numpy()
    grp = prim["chem_class"].to_numpy()
    logo = LeaveOneGroupOut()
    model = make_models()["gbm"]
    rows = []
    for tr, te in logo.split(X, y, grp):
        cls = grp[te][0]
        if len(np.unique(y[te])) < 2 or len(te) < 50:
            continue
        model.fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        rows.append(
            {
                "held_out_class": cls,
                "n": len(te),
                "auc": roc_auc_score(y[te], p),
                "obs_agree": y[te].mean(),
            }
        )
    return pd.DataFrame(rows)


def permutation_importance_auc(prim, oof_model="gbm"):
    """Simple grouped permutation importance on a single fit (interpretability)."""
    X = prim[NUM + CAT].copy()
    y = prim["agree"].to_numpy()
    groups = prim["chem_bare"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    tr, te = next(gkf.split(X, y, groups))
    model = make_models()[oof_model]
    model.fit(X.iloc[tr], y[tr])
    base = roc_auc_score(y[te], model.predict_proba(X.iloc[te])[:, 1])
    rng = np.random.default_rng(0)
    rows = []
    Xte = X.iloc[te].reset_index(drop=True)
    yte = y[te]
    for col in NUM + CAT:
        drops = []
        for _ in range(5):
            Xp = Xte.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - roc_auc_score(yte, model.predict_proba(Xp)[:, 1]))
        rows.append({"feature": col, "auc_drop": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False), base


def main():
    cfg = Config.default()
    df = pd.read_parquet(cfg.derived / "effect_keys.parquet")
    prim = build_modeling_frame(df)
    prim.to_parquet(cfg.derived / "aim3_modeling_frame.parquet", index=False)
    print(f"Modeling rows: {len(prim):,}  agree rate: {prim['agree'].mean():.3f}")

    cv, oof = grouped_cv(prim, cfg)
    cv.to_csv(cfg.tables / "A3_cv_performance.csv", index=False)
    print("\n=== Grouped (by chemical) 5-fold CV ===")
    print(cv.to_string(index=False))

    cal = calibration_table(prim["agree"].to_numpy(), oof["gbm"])
    cal.to_csv(cfg.tables / "A3_calibration_gbm.csv", index=False)
    print("\n=== Calibration (GBM, out-of-fold) ===")
    print(cal.to_string(index=False))

    loco = loco_generalization(prim)
    loco.to_csv(cfg.tables / "A3_leave_one_class_out.csv", index=False)
    print("\n=== Leave-one-chemical-class-out generalization (GBM) ===")
    print(loco.to_string(index=False))

    imp, base_auc = permutation_importance_auc(prim)
    imp.to_csv(cfg.tables / "A3_permutation_importance.csv", index=False)
    print(f"\n=== Permutation importance (GBM, fold AUC={base_auc:.3f}) ===")
    print(imp.to_string(index=False))

    print("\nDONE Aim 3")


if __name__ == "__main__":
    main()
