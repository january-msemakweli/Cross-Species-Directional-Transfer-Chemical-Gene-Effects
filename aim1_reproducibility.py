"""
Aim 1: cross-species directional reproducibility (PMID-disjoint primary).

Estimand: P(human consensus direction == rodent consensus direction | both
species have an unambiguous consensus, disjoint PMIDs), overall and stratified.
Inference: cluster bootstrap (clusters = chemical) plus naive Wilson for scale.
Also: mouse-vs-rat agreement as an internal reproducibility ceiling, and a
chance-agreement baseline from directional marginals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_core import Config, cluster_bootstrap_mean, wilson_ci


def agreement_frame(df: pd.DataFrame, a_col: str, b_col: str) -> pd.DataFrame:
    sub = df[(df[a_col] != 0) & (df[b_col] != 0)].copy()
    sub["agree"] = (sub[a_col] == sub[b_col]).astype(int)
    return sub


def summarize(sub: pd.DataFrame, cfg: Config, label: str) -> dict:
    n = len(sub)
    if n == 0:
        return {"stratum": label, "n": 0}
    k = int(sub["agree"].sum())
    p, wlo, whi = wilson_ci(k, n)
    _, blo, bhi = cluster_bootstrap_mean(
        sub["agree"].to_numpy(), sub["chem_bare"].to_numpy(), cfg.n_boot, cfg.seed
    )
    # chance baseline from human marginal of +1 vs -1 in this subset
    p_h_inc = (sub[sub.columns[sub.columns.get_loc("h_cons")]] == 1).mean()
    p_r_inc = (sub["rod_cons"] == 1).mean() if "rod_cons" in sub else np.nan
    chance = p_h_inc * p_r_inc + (1 - p_h_inc) * (1 - p_r_inc)
    return {
        "stratum": label,
        "n": n,
        "agree": k,
        "p_agree": p,
        "wilson_lo": wlo,
        "wilson_hi": whi,
        "boot_lo": blo,
        "boot_hi": bhi,
        "chance_agree": float(chance),
        "excess_over_chance": float(p - chance) if np.isfinite(chance) else np.nan,
    }


def main():
    cfg = Config.default()
    df = pd.read_parquet(cfg.derived / "effect_keys.parquet")
    both = df[df["in_human"] & df["in_rodent"]].copy()

    # Primary: PMID-disjoint, unambiguous both
    prim = both[both["pmid_disjoint_hr"]].copy()
    prim_ag = agreement_frame(prim, "h_cons", "rod_cons")

    rows = [summarize(prim_ag, cfg, "PRIMARY_human_vs_rodent_disjoint")]

    # Secondary: include shared-PMID pairs (shows inflation from co-curation)
    all_ag = agreement_frame(both, "h_cons", "rod_cons")
    rows.append(summarize(all_ag, cfg, "SECONDARY_human_vs_rodent_all"))
    shared_ag = agreement_frame(both[~both["pmid_disjoint_hr"]], "h_cons", "rod_cons")
    rows.append(summarize(shared_ag, cfg, "AUX_human_vs_rodent_sharedPMID"))

    # Internal ceiling: mouse vs rat (two rodents, disjoint by construction rarely)
    mr = df[df["in_mouse"] & df["in_rat"]].copy()
    mr_ag = agreement_frame(mr, "m_cons", "r_cons")
    mr_ag = mr_ag.rename(columns={})  # chem_bare present
    rows.append(summarize(mr_ag.assign(rod_cons=mr_ag["r_cons"]), cfg, "CEILING_mouse_vs_rat"))

    overall = pd.DataFrame(rows)
    overall.to_csv(cfg.tables / "A1_overall_agreement.csv", index=False)
    print("=== Aim 1 overall ===")
    print(overall[["stratum", "n", "p_agree", "boot_lo", "boot_hi", "chance_agree", "excess_over_chance"]].to_string(index=False))

    # Stratified by aspect (primary set)
    by_aspect = []
    for asp, g in prim_ag.groupby("aspect"):
        if len(g) < cfg.min_stratum:
            continue
        by_aspect.append(summarize(g, cfg, asp))
    by_aspect = pd.DataFrame(by_aspect).sort_values("n", ascending=False)
    by_aspect.to_csv(cfg.tables / "A1_by_aspect.csv", index=False)
    print("\n=== Aim 1 by molecular aspect (primary) ===")
    print(by_aspect[["stratum", "n", "p_agree", "boot_lo", "boot_hi", "excess_over_chance"]].to_string(index=False))

    # Stratified by chemical class
    by_class = []
    for cls, g in prim_ag.groupby("chem_class"):
        if len(g) < cfg.min_stratum:
            continue
        by_class.append(summarize(g, cfg, cls))
    by_class = pd.DataFrame(by_class).sort_values("n", ascending=False)
    by_class.to_csv(cfg.tables / "A1_by_chem_class.csv", index=False)
    print("\n=== Aim 1 by chemical class (primary) ===")
    print(by_class[["stratum", "n", "p_agree", "boot_lo", "boot_hi", "excess_over_chance"]].to_string(index=False))

    # Effect of evidence depth: agreement vs min supporting PMIDs
    prim_ag = prim_ag.assign(
        depth=lambda d: np.minimum(d["h_pmids"], d["rod_pmids"]).clip(upper=6)
    )
    depth_tab = []
    for depth, g in prim_ag.groupby("depth"):
        if len(g) < cfg.min_stratum:
            continue
        depth_tab.append(summarize(g, cfg, f"min_pmids={depth}"))
    depth_tab = pd.DataFrame(depth_tab)
    depth_tab.to_csv(cfg.tables / "A1_by_depth.csv", index=False)
    print("\n=== Aim 1 by evidence depth (min PMIDs per side) ===")
    print(depth_tab[["stratum", "n", "p_agree", "boot_lo", "boot_hi"]].to_string(index=False))

    print("\nDONE Aim 1")


if __name__ == "__main__":
    main()
