"""
Aim 4: external anchoring of directional transferability.

Pre-registered anchor (chosen before modeling): evolutionary conservation of
the human gene to its rodent ortholog, taken from a source OUTSIDE CTD curation:
  - Ensembl Compara human->mouse and human->rat protein % identity
    (graded: `perc_id_rodent_mean`, `perc_id_rodent_min`), and
  - 1:1 orthology status (Ensembl `ortholog_one2one` for both rodents).
A secondary anchor is NCBI gene_orthologs 1:1 status.

Hypothesis (directional, pre-specified): effects on more conserved genes
transfer more reliably from rodent to human, i.e. directional agreement
(PMID-disjoint, both species unambiguous) increases with conservation.

Because orthology presence is near-universal among CTD-studied genes, the
informative axis is the GRADED % identity, not mere ortholog presence.

Outputs (tables/):
  A4_coverage.csv                external-anchor coverage of the primary set
  A4_agreement_by_conservation.csv   agreement by %id quartile (cluster boot)
  A4_conservation_trend.csv      cluster-robust logistic trend (OR per +10 %id)
  A4_orthology_one2one.csv       1:1 vs not (odds ratio)
  A4_incremental_cv.csv          does conservation add over CTD-internal model?
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from analysis_core import Config, cluster_bootstrap_mean, two_by_two, wilson_ci
from aim3_prediction import CAT, NUM, build_modeling_frame, make_models

def _external_dir() -> Path:
    """Prefer ./external_data (public-repo layout), else ../external_data."""
    root = Path(__file__).resolve().parent
    for cand in (root / "external_data", root.parent / "external_data"):
        if (cand / "ensembl_conservation_hmr.parquet").exists():
            return cand
    return root / "external_data"


EXTERNAL = _external_dir()


def load_conservation() -> pd.DataFrame:
    """Ensembl graded conservation + NCBI 1:1 status, keyed by Entrez GeneID."""
    ens = pd.read_parquet(EXTERNAL / "ensembl_conservation_hmr.parquet")
    ens["gene"] = ens["gene"].astype(str)
    ens["ens_1to1_both"] = (
        ens["is_1to1_mouse"].fillna(False) & ens["is_1to1_rat"].fillna(False)
    ).astype(int)

    ncbi = pd.read_parquet(EXTERNAL / "human_mouse_rat_orthologs.parquet")
    ncbi = ncbi.rename(columns={"human_geneid": "gene"})
    ncbi["gene"] = ncbi["gene"].astype(str)
    ncbi = ncbi[["gene", "has_both_rodents", "is_1to1_hmr"]].rename(
        columns={"has_both_rodents": "ncbi_both", "is_1to1_hmr": "ncbi_1to1"}
    )

    cons = ens.merge(ncbi, on="gene", how="outer")
    return cons


def attach(prim: pd.DataFrame, cons: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "gene", "perc_id_mouse", "perc_id_rat", "perc_id_rodent_mean",
        "perc_id_rodent_min", "ens_1to1_both", "ncbi_both", "ncbi_1to1",
    ]
    m = prim.merge(cons[keep], on="gene", how="left")
    return m


# ---------------------------------------------------------------------------
# (B) agreement by conservation quartile
# ---------------------------------------------------------------------------
def agreement_by_conservation(m: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sub = m[m["perc_id_rodent_mean"].notna()].copy()
    sub["cons_q"] = pd.qcut(sub["perc_id_rodent_mean"], 4, labels=False, duplicates="drop")
    rows = []
    for q, g in sub.groupby("cons_q"):
        n = len(g)
        k = int(g["agree"].sum())
        p, wlo, whi = wilson_ci(k, n)
        _, blo, bhi = cluster_bootstrap_mean(
            g["agree"].to_numpy(), g["chem_bare"].to_numpy(), cfg.n_boot, cfg.seed
        )
        rows.append({
            "cons_quartile": int(q) + 1,
            "perc_id_lo": float(g["perc_id_rodent_mean"].min()),
            "perc_id_hi": float(g["perc_id_rodent_mean"].max()),
            "perc_id_median": float(g["perc_id_rodent_mean"].median()),
            "n": n, "agree": k, "p_agree": p,
            "wilson_lo": wlo, "wilson_hi": whi,
            "boot_lo": blo, "boot_hi": bhi,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (C) cluster-robust logistic trend
# ---------------------------------------------------------------------------
def conservation_trend(m: pd.DataFrame) -> pd.DataFrame:
    sub = m[m["perc_id_rodent_mean"].notna()].copy()
    x = sub["perc_id_rodent_mean"].to_numpy(float)
    X = sm.add_constant(x.reshape(-1, 1))
    y = sub["agree"].to_numpy(float)
    groups = sub["chem_bare"].astype("category").cat.codes.to_numpy()
    model = sm.GLM(y, X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    beta = res.params[1]
    ci = res.conf_int()[1]
    # OR per +10 percentage points identity
    def to_or(b):
        return float(np.exp(b * 10.0))
    out = pd.DataFrame([{
        "term": "perc_id_rodent_mean",
        "beta_per_pct": float(beta),
        "p_value": float(res.pvalues[1]),
        "or_per_10pct": to_or(beta),
        "or_lo_per_10pct": to_or(ci[0]),
        "or_hi_per_10pct": to_or(ci[1]),
        "n": int(len(sub)),
        "n_chemicals": int(sub["chem_bare"].nunique()),
    }])
    return out


# ---------------------------------------------------------------------------
# (D) 1:1 orthology contrast
# ---------------------------------------------------------------------------
def orthology_contrast(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for flag, label in [("ens_1to1_both", "ensembl_one2one_both"),
                        ("ncbi_1to1", "ncbi_1to1")]:
        sub = m[m[flag].notna()].copy()
        sub[flag] = sub[flag].astype(int)
        a = int(((sub[flag] == 1) & (sub["agree"] == 1)).sum())  # 1:1 & agree
        b = int(((sub[flag] == 1) & (sub["agree"] == 0)).sum())  # 1:1 & disagree
        c = int(((sub[flag] == 0) & (sub["agree"] == 1)).sum())  # not & agree
        d = int(((sub[flag] == 0) & (sub["agree"] == 0)).sum())  # not & disagree
        stats = two_by_two(a, b, c, d)
        p1 = a / (a + b) if (a + b) else np.nan
        p0 = c / (c + d) if (c + d) else np.nan
        rows.append({
            "anchor": label,
            "n_1to1": a + b, "agree_1to1": p1,
            "n_not": c + d, "agree_not": p0,
            "odds_ratio": stats["or"], "or_lo": stats["or_lo"],
            "or_hi": stats["or_hi"], "fisher_p": stats["fisher_p"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (E) incremental predictive value over the CTD-internal model
# ---------------------------------------------------------------------------
CONS_NUM = ["perc_id_rodent_mean", "perc_id_rodent_min"]


def incremental_cv(m: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Grouped-by-chemical CV: base (CTD-internal) vs base + conservation.
    Restricted to rows with a conservation value so the comparison is paired."""
    sub = m[m["perc_id_rodent_mean"].notna()].copy()
    # median-impute the min (few NaN) so both feature sets see identical rows
    sub["perc_id_rodent_min"] = sub["perc_id_rodent_min"].fillna(
        sub["perc_id_rodent_min"].median()
    )
    y = sub["agree"].to_numpy()
    groups = sub["chem_bare"].to_numpy()
    gkf = GroupKFold(n_splits=5)

    def run(num_features):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder

        pre = ColumnTransformer([
            ("num", "passthrough", num_features),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT),
        ])
        clf = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=250,
            l2_regularization=1.0, early_stopping=True, n_iter_no_change=15,
            random_state=0,
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        X = sub[num_features + CAT]
        aucs, briers = [], []
        oof = np.full(len(y), np.nan)
        for tr, te in gkf.split(X, y, groups):
            pipe.fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[te])[:, 1]
            oof[te] = p
            aucs.append(roc_auc_score(y[te], p))
            briers.append(brier_score_loss(y[te], p))
        return np.array(aucs), np.array(briers), oof

    base_auc, base_brier, _ = run(NUM)
    full_auc, full_brier, _ = run(NUM + CONS_NUM)
    d_auc = full_auc - base_auc  # paired across folds
    rows = [
        {"model": "ctd_internal", "auc_mean": base_auc.mean(),
         "auc_sd": base_auc.std(), "brier_mean": base_brier.mean()},
        {"model": "ctd_internal+conservation", "auc_mean": full_auc.mean(),
         "auc_sd": full_auc.std(), "brier_mean": full_brier.mean()},
        {"model": "delta(full-base)", "auc_mean": d_auc.mean(),
         "auc_sd": d_auc.std(),
         "brier_mean": (full_brier - base_brier).mean()},
    ]
    return pd.DataFrame(rows), sub

def main():
    cfg = Config.default()
    df = pd.read_parquet(cfg.derived / "effect_keys.parquet")
    prim = build_modeling_frame(df)
    cons = load_conservation()
    m = attach(prim, cons)

    # (A) coverage
    n = len(m)
    cov = pd.DataFrame([{
        "n_primary_keys": n,
        "with_ensembl_percid": int(m["perc_id_rodent_mean"].notna().sum()),
        "frac_ensembl": float(m["perc_id_rodent_mean"].notna().mean()),
        "with_ncbi_ortholog": int(m["ncbi_both"].notna().sum()),
        "median_perc_id": float(m["perc_id_rodent_mean"].median()),
    }])
    cov.to_csv(cfg.tables / "A4_coverage.csv", index=False)
    m.to_parquet(cfg.derived / "aim4_modeling_frame.parquet", index=False)
    print("=== Aim 4 coverage ===")
    print(cov.to_string(index=False))

    # (B) agreement by conservation quartile
    byq = agreement_by_conservation(m, cfg)
    byq.to_csv(cfg.tables / "A4_agreement_by_conservation.csv", index=False)
    print("\n=== Agreement by conservation quartile (Ensembl %id) ===")
    print(byq.to_string(index=False))

    # (C) cluster-robust logistic trend
    trend = conservation_trend(m)
    trend.to_csv(cfg.tables / "A4_conservation_trend.csv", index=False)
    print("\n=== Conservation trend (cluster-robust logistic) ===")
    print(trend.to_string(index=False))

    # (D) 1:1 orthology contrast
    orth = orthology_contrast(m)
    orth.to_csv(cfg.tables / "A4_orthology_one2one.csv", index=False)
    print("\n=== 1:1 orthology contrast ===")
    print(orth.to_string(index=False))

    # (E) incremental predictive value
    inc, _ = incremental_cv(m, cfg)
    inc.to_csv(cfg.tables / "A4_incremental_cv.csv", index=False)
    print("\n=== Incremental value of conservation over CTD-internal model ===")
    print(inc.to_string(index=False))

    print("\nDONE Aim 4")


if __name__ == "__main__":
    main()
