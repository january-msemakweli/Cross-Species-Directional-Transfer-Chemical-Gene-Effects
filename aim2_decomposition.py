"""
Aim 2: decompose directional non-agreement into
  (i)  within-species inconsistency (context/noise: same chem-gene curated both
       up and down inside one species),
  (ii) same-clade cross-species divergence (mouse vs rat),
  (iii) cross-clade divergence (human vs rodent).

Key inferential target: the *human-specific* divergence component
    = disagreement(human vs rodent) - disagreement(mouse vs rat),
bootstrapped over chemicals. If ~0, phylogeny adds little beyond general
irreproducibility, and the reproducibility ceiling (not species) is the story.

Plus a GLMM variance-components model (random intercepts: chemical, gene, aspect)
on the binary agreement outcome, and a ranked catalog of hard cross-species
directional conflicts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_core import Config


def within_species_inconsistency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sp, inc, dec, seen in [
        ("human", "h_inc", "h_dec", "in_human"),
        ("mouse", "m_inc", "m_dec", "in_mouse"),
        ("rat", "r_inc", "r_dec", "in_rat"),
    ]:
        s = df[df[seen]]
        multi = s[(s[inc] + s[dec]) >= 2]
        amb = ((multi[inc] > 0) & (multi[dec] > 0)).mean() if len(multi) else np.nan
        rows.append(
            {
                "species": sp,
                "n_keys_seen": int(s[seen].sum()),
                "n_keys_multi_row": len(multi),
                "within_ambiguity_rate": float(amb),
            }
        )
    return pd.DataFrame(rows)


def disagreement(df, a, b):
    sub = df[(df[a] != 0) & (df[b] != 0)]
    if len(sub) == 0:
        return np.nan, 0
    return float((sub[a] != sub[b]).mean()), len(sub)


def bootstrap_human_specific(df: pd.DataFrame, cfg: Config) -> dict:
    """Cluster (chemical) bootstrap of disagree(H,R) - disagree(M,R)."""
    rng = np.random.default_rng(cfg.seed)
    hr = df[df["in_human"] & df["in_rodent"] & df["pmid_disjoint_hr"]].copy()
    mr = df[df["in_mouse"] & df["in_rat"]].copy()
    hr = hr[(hr["h_cons"] != 0) & (hr["rod_cons"] != 0)]
    mr = mr[(mr["m_cons"] != 0) & (mr["r_cons"] != 0)]
    hr_dis = (hr["h_cons"] != hr["rod_cons"]).to_numpy().astype(float)
    mr_dis = (mr["m_cons"] != mr["r_cons"]).to_numpy().astype(float)
    hr_chem = hr["chem_bare"].to_numpy()
    mr_chem = mr["chem_bare"].to_numpy()
    obs = hr_dis.mean() - mr_dis.mean()

    # resample chemicals shared across both, recompute
    chems = np.union1d(hr_chem, mr_chem)
    hr_by = {c: hr_dis[hr_chem == c] for c in np.unique(hr_chem)}
    mr_by = {c: mr_dis[mr_chem == c] for c in np.unique(mr_chem)}
    diffs = []
    for _ in range(cfg.n_boot):
        pick = rng.choice(chems, size=len(chems), replace=True)
        h = np.concatenate([hr_by[c] for c in pick if c in hr_by]) if any(c in hr_by for c in pick) else np.array([])
        m = np.concatenate([mr_by[c] for c in pick if c in mr_by]) if any(c in mr_by for c in pick) else np.array([])
        if len(h) and len(m):
            diffs.append(h.mean() - m.mean())
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "disagree_human_rodent": float(hr_dis.mean()),
        "disagree_mouse_rat": float(mr_dis.mean()),
        "human_specific_divergence": float(obs),
        "hsd_boot_lo": float(lo),
        "hsd_boot_hi": float(hi),
        "n_hr": len(hr_dis),
        "n_mr": len(mr_dis),
    }


def _icc_oneway(values: np.ndarray, groups: np.ndarray) -> dict:
    """One-way random-effects ANOVA ICC (method of moments) for a 0/1 outcome.

    Estimates the share of agreement variance attributable to the grouping factor.
    Uses the standard unbiased between/within decomposition with an n0 correction
    for unequal group sizes. ICC is clipped to [0, 1].
    """
    values = np.asarray(values, dtype=float)
    df = pd.DataFrame({"y": values, "g": groups})
    grp = df.groupby("g")["y"]
    ni = grp.size().to_numpy()
    mi = grp.mean().to_numpy()
    N = values.size
    k = len(ni)
    if k < 2 or N <= k:
        return {"icc": np.nan, "n_groups": k, "var_between": np.nan, "var_within": np.nan}
    grand = values.mean()
    ss_between = float(np.sum(ni * (mi - grand) ** 2))
    ss_within = float(np.sum((values - df.groupby("g")["y"].transform("mean")) ** 2))
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (N - k)
    n0 = (N - np.sum(ni**2) / N) / (k - 1)
    var_between = max(0.0, (ms_between - ms_within) / n0)
    var_within = ms_within
    icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else np.nan
    return {
        "icc": float(np.clip(icc, 0, 1)),
        "n_groups": int(k),
        "var_between": float(var_between),
        "var_within": float(var_within),
    }


def variance_components(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Where does agreement variance concentrate? ICC by chemical, gene, aspect,
    and chemical-gene pair, via one-way random-effects ANOVA on the 0/1 agree
    outcome (fast, model-light, no dense design matrices)."""
    prim = df[df["in_human"] & df["in_rodent"] & df["pmid_disjoint_hr"]].copy()
    prim = prim[(prim["h_cons"] != 0) & (prim["rod_cons"] != 0)]
    prim["agree"] = (prim["h_cons"] == prim["rod_cons"]).astype(int)
    prim["chem_gene"] = prim["chem_bare"].astype(str) + "|" + prim["gene"].astype(str)
    y = prim["agree"].to_numpy()
    out = []
    for factor, col in [
        ("chemical", "chem_bare"),
        ("gene", "gene"),
        ("aspect", "aspect"),
        ("chem_gene_pair", "chem_gene"),
        ("chem_class", "chem_class"),
    ]:
        r = _icc_oneway(y, prim[col].to_numpy())
        r["factor"] = factor
        out.append(r)
    res = pd.DataFrame(out)[["factor", "n_groups", "icc", "var_between", "var_within"]]
    return res


def conflict_catalog(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Strong, independent human-vs-rodent directional conflicts: candidate true
    divergences or curation errors. Require multi-PMID support on both sides."""
    c = df[
        df["in_human"]
        & df["in_rodent"]
        & df["pmid_disjoint_hr"]
        & (df["h_cons"] != 0)
        & (df["rod_cons"] != 0)
        & (df["h_cons"] != df["rod_cons"])
    ].copy()
    c["h_support"] = c[["h_inc", "h_dec"]].max(axis=1)
    c["rod_support"] = c[["rod_inc", "rod_dec"]].max(axis=1)
    c["min_support"] = c[["h_support", "rod_support"]].min(axis=1)
    c["strength"] = c["h_pmids"].clip(upper=10) + c["rod_pmids"].clip(upper=10)
    strong = c[c["min_support"] >= 2].sort_values(
        ["strength", "min_support"], ascending=False
    )
    cols = [
        "chem_name", "gene_symbol", "aspect", "h_cons", "rod_cons",
        "h_inc", "h_dec", "rod_inc", "rod_dec", "h_pmids", "rod_pmids", "chem_class",
    ]
    return strong[cols]


def main():
    cfg = Config.default()
    df = pd.read_parquet(cfg.derived / "effect_keys.parquet")

    wi = within_species_inconsistency(df)
    wi.to_csv(cfg.tables / "A2_within_species_inconsistency.csv", index=False)
    print("=== Within-species inconsistency (context/noise) ===")
    print(wi.to_string(index=False))

    hsd = bootstrap_human_specific(df, cfg)
    pd.DataFrame([hsd]).to_csv(cfg.tables / "A2_human_specific_divergence.csv", index=False)
    print("\n=== Clade decomposition ===")
    print(f"disagree(human,rodent) = {hsd['disagree_human_rodent']:.3f}  (n={hsd['n_hr']:,})")
    print(f"disagree(mouse,rat)    = {hsd['disagree_mouse_rat']:.3f}  (n={hsd['n_mr']:,})")
    print(
        f"HUMAN-SPECIFIC divergence = {hsd['human_specific_divergence']:.3f} "
        f"(95% CI {hsd['hsd_boot_lo']:.3f} to {hsd['hsd_boot_hi']:.3f})"
    )
    print(
        "Interpretation: if this CI includes ~0 or is small, most cross-species "
        "'failure' is general irreproducibility (context/noise), not phylogeny."
    )

    vc = variance_components(df, cfg)
    vc.to_csv(cfg.tables / "A2_variance_components_icc.csv", index=False)
    print("\n=== Variance components: ICC of agreement by factor ===")
    print(vc.to_string(index=False))

    cat = conflict_catalog(df, cfg)
    cat.to_csv(cfg.tables / "A2_conflict_catalog.csv", index=False)
    print(f"\n=== Hard cross-species conflicts (>=2 supporting rows/side): {len(cat):,} ===")
    print(cat.head(20).to_string(index=False))

    print("\nDONE Aim 2")


if __name__ == "__main__":
    main()
