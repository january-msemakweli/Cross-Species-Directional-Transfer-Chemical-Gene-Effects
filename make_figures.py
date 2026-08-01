"""
Figures for the cross-species directional-transfer study, in the Lancet-style
house aesthetic (see figstyle.py). Reads the CSV tables produced by the aim
scripts; writes PNG + PDF + TIF to figures/.
"""
from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_core import Config
from figstyle import (
    CAT_COLORS,
    CMAP_SEQ,
    PALETTE,
    apply_style,
    panel_label,
    ringed,
    save_fig,
    style_panel,
)

apply_style()


def _errbars(p, lo, hi):
    return np.clip(p - lo, 0, None), np.clip(hi - p, 0, None)


def fig_overview(cfg: Config):
    ov = pd.read_csv(cfg.tables / "A1_overall_agreement.csv")
    order = [
        "AUX_human_vs_rodent_sharedPMID",
        "SECONDARY_human_vs_rodent_all",
        "CEILING_mouse_vs_rat",
        "PRIMARY_human_vs_rodent_disjoint",
    ]
    labels = {
        "AUX_human_vs_rodent_sharedPMID": "Human vs rodent\n(shared PMIDs)",
        "SECONDARY_human_vs_rodent_all": "Human vs rodent\n(all pairs)",
        "CEILING_mouse_vs_rat": "Mouse vs rat\n(within-rodent benchmark)",
        "PRIMARY_human_vs_rodent_disjoint": "Human vs rodent\n(independent, PRIMARY)",
    }
    colors = {
        "AUX_human_vs_rodent_sharedPMID": PALETTE["crimson"],
        "SECONDARY_human_vs_rodent_all": PALETTE["orange"],
        "CEILING_mouse_vs_rat": PALETTE["green"],
        "PRIMARY_human_vs_rodent_disjoint": PALETTE["navy"],
    }
    ov = ov.set_index("stratum").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    style_panel(ax, grid_axis="x")
    y = np.arange(len(ov))
    p = ov["p_agree"].to_numpy()
    lo, hi = _errbars(p, ov["boot_lo"].to_numpy(), ov["boot_hi"].to_numpy())
    bar_colors = [colors[s] for s in ov["stratum"]]
    ax.barh(y, p, color=bar_colors, alpha=0.9, height=0.62, zorder=2,
            edgecolor="white", linewidth=1.0)
    ax.errorbar(p, y, xerr=[lo, hi], fmt="none", ecolor=PALETTE["ink"],
                capsize=4, lw=1.4, zorder=4)
    ax.scatter(ov["chance_agree"], y, marker="D", s=55, color="white",
               edgecolors=PALETTE["ink"], linewidths=1.4, zorder=5, label="chance baseline")
    for i, row in ov.iterrows():
        ax.text(min(row["p_agree"] + hi[i] + 0.015, 0.965), i, f"{row['p_agree']*100:.0f}%",
                va="center", ha="left", fontsize=10.5, fontweight="bold",
                color=colors[row["stratum"]])
    ax.axvline(0.5, ls=":", color=PALETTE["muted"], lw=1.2, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[s] for s in ov["stratum"]])
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("Directional agreement")
    leg = ax.legend(loc="upper right", handletextpad=0.4, frameon=True, fancybox=True,
                    framealpha=0.95, edgecolor="#D0D5DD", borderpad=0.6)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    save_fig(fig, "F1_overview_agreement", cfg.figures)


def fig_by_aspect(cfg: Config):
    a = pd.read_csv(cfg.tables / "A1_by_aspect.csv").sort_values("p_agree").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    style_panel(ax, grid_axis="x")
    y = np.arange(len(a))
    p = a["p_agree"].to_numpy()
    lo, hi = _errbars(p, a["boot_lo"].to_numpy(), a["boot_hi"].to_numpy())
    norm = (p - 0.4) / (1.0 - 0.4)
    mcolors = CMAP_SEQ(np.clip(norm, 0, 1))
    for i in range(len(a)):
        ax.plot([0.5, p[i]], [i, i], color=PALETTE["muted"], lw=1.2, zorder=1, alpha=0.6)
    ax.errorbar(p, y, xerr=[lo, hi], fmt="none", ecolor=PALETTE["ink"], capsize=3, lw=1.2, zorder=3)
    ax.scatter(p, y, s=95, c=mcolors, edgecolors="white", linewidths=1.6, zorder=4)
    for i, row in a.iterrows():
        ax.text(1.075, i, f"n={int(row['n']):,}", va="center", ha="right",
                fontsize=8.5, color=PALETTE["muted"])
    ax.axvline(0.5, ls=":", color=PALETTE["muted"], lw=1.2, label="chance", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(a["stratum"])
    ax.set_xlim(0.3, 1.08)
    ax.set_xlabel("Directional agreement (independent, human vs rodent)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99))
    fig.tight_layout()
    save_fig(fig, "F2_by_aspect", cfg.figures)


def fig_depth(cfg: Config):
    d = pd.read_csv(cfg.tables / "A1_by_depth.csv")
    d["k"] = d["stratum"].str.replace("min_pmids=", "").astype(int)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    style_panel(ax, grid_axis="y")
    p = d["p_agree"].to_numpy()
    lo, hi = _errbars(p, d["boot_lo"].to_numpy(), d["boot_hi"].to_numpy())
    col = PALETTE["crimson"]
    ax.plot(d["k"].to_numpy(), p, "-", color=col, lw=2.8, zorder=3, solid_capstyle="round")
    ax.errorbar(d["k"], p, yerr=[lo, hi], fmt="none", ecolor=PALETTE["ink"], capsize=3, lw=1.3, zorder=4)
    ringed(ax, d["k"].to_numpy(), p, col, s=80)
    for xi, yi, n in zip(d["k"], p, d["n"]):
        dx, ha = (12, "left") if xi < d["k"].max() else (-12, "right")
        ax.annotate(f"n={int(n):,}", (xi, yi), textcoords="offset points",
                    xytext=(dx, -6), ha=ha, va="top", fontsize=8.5,
                    color=PALETTE["ink"], zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    ax.axhline(0.5, ls=":", color=PALETTE["muted"], lw=1.2)
    ax.set_xlabel("Minimum independent PubMed IDs per species")
    ax.set_ylabel("Directional agreement")
    ax.set_ylim(0.45, 1.03)
    fig.tight_layout()
    save_fig(fig, "F3_depth", cfg.figures)


def fig_decomposition(cfg: Config):
    hsd = pd.read_csv(cfg.tables / "A2_human_specific_divergence.csv").iloc[0]
    icc = pd.read_csv(cfg.tables / "A2_variance_components_icc.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))

    ax = axes[0]
    style_panel(ax, grid_axis="y")
    cats = ["Mouse vs rat\n(same clade)", "Human vs rodent\n(cross clade)"]
    vals = [hsd["disagree_mouse_rat"], hsd["disagree_human_rodent"]]
    ax.bar(cats, vals, color=[PALETTE["green"], PALETTE["navy"]], alpha=0.9,
           width=0.55, edgecolor="white", linewidth=1.2, zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v*100:.1f}%", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=PALETTE["ink"])
    ax.set_ylabel("Directional DISagreement rate")
    ax.set_ylim(0, 0.5)
    ax.annotate(
        f"human-specific\n= {hsd['human_specific_divergence']*100:.1f} pts\n"
        f"(95% CI {hsd['hsd_boot_lo']*100:.1f} to {hsd['hsd_boot_hi']*100:.1f})",
        xy=(1, vals[1]), xytext=(0.28, 0.44),
        arrowprops=dict(arrowstyle="->", color=PALETTE["ink"], lw=1.2), fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D0D5DD", alpha=0.95),
    )
    panel_label(ax, "A", "Directional disagreement")

    ax = axes[1]
    style_panel(ax, grid_axis="x")
    icc = icc.sort_values("icc").reset_index(drop=True)
    y = np.arange(len(icc))
    ax.barh(y, icc["icc"], color=PALETTE["crimson"], alpha=0.9, height=0.6,
            edgecolor="white", linewidth=1.0, zorder=2)
    for i, r in icc.iterrows():
        ax.text(r["icc"] + 0.008, i, f"{r['icc']:.2f}", va="center", fontsize=10, color=PALETTE["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(icc["factor"])
    ax.set_xlim(0, max(icc["icc"]) * 1.18)
    ax.set_xlabel("ICC (share of agreement variance)")
    panel_label(ax, "B", "Variance components")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "F4_decomposition", cfg.figures)


def fig_prediction(cfg: Config):
    cal = pd.read_csv(cfg.tables / "A3_calibration_gbm.csv")
    imp = pd.read_csv(cfg.tables / "A3_permutation_importance.csv")
    cv = pd.read_csv(cfg.tables / "A3_cv_performance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))

    ax = axes[0]
    style_panel(ax)
    lo, hi = 0.4, 0.95
    ax.plot([lo, hi], [lo, hi], ls="--", color=PALETTE["muted"], lw=1.4, label="perfect", zorder=2)
    ax.plot(cal["pred"].to_numpy(), cal["obs"].to_numpy(), "-", color=PALETTE["navy"],
            lw=2.8, zorder=3, solid_capstyle="round")
    ringed(ax, cal["pred"].to_numpy(), cal["obs"].to_numpy(), PALETTE["navy"], s=62)
    gbm_auc = cv.loc[cv["model"] == "gbm", "auc_mean"].iloc[0]
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted P(transfer)")
    ax.set_ylabel("Observed agreement")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.05, f"grouped-CV AUC = {gbm_auc:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5, color=PALETTE["ink"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D0D5DD", alpha=0.95))
    panel_label(ax, "A", "Calibration")

    ax = axes[1]
    style_panel(ax, grid_axis="x")
    imp = imp.sort_values("auc_drop").reset_index(drop=True)
    y = np.arange(len(imp))
    bar_colors = [PALETTE["green"] if v >= 0 else PALETTE["crimson"] for v in imp["auc_drop"]]
    ax.barh(y, imp["auc_drop"], color=bar_colors, alpha=0.9, height=0.6,
            edgecolor="white", linewidth=1.0, zorder=2)
    ax.axvline(0, color=PALETTE["ink"], lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(imp["feature"])
    ax.set_xlabel("AUC drop when permuted")
    panel_label(ax, "B", "Permutation importance")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "F5_prediction", cfg.figures)


def fig_conservation(cfg: Config):
    """Aim 4: external conservation anchor. (A) agreement vs %id quartile is
    flat; (B) forest of external anchors, all straddling OR = 1."""
    byq = pd.read_csv(cfg.tables / "A4_agreement_by_conservation.csv")
    trend = pd.read_csv(cfg.tables / "A4_conservation_trend.csv").iloc[0]
    orth = pd.read_csv(cfg.tables / "A4_orthology_one2one.csv").set_index("anchor")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7))

    # Panel A: agreement vs conservation quartile
    ax = axes[0]
    style_panel(ax, grid_axis="y")
    x = byq["perc_id_median"].to_numpy()
    p = byq["p_agree"].to_numpy()
    lo, hi = _errbars(p, byq["boot_lo"].to_numpy(), byq["boot_hi"].to_numpy())
    ax.errorbar(x, p, yerr=[lo, hi], fmt="none", ecolor=PALETTE["ink"],
                capsize=4, lw=1.4, zorder=3)
    ringed(ax, x, p, PALETTE["navy"], s=90)
    overall = float((byq["agree"].sum()) / (byq["n"].sum()))
    ax.axhline(overall, ls=":", color=PALETTE["muted"], lw=1.4,
               label=f"pooled = {overall*100:.0f}%")
    for xi, n in zip(x, byq["n"]):
        ax.annotate(f"n={int(n):,}", (xi, 0.508), ha="center", va="bottom",
                    fontsize=8.5, color=PALETTE["muted"], zorder=6)
    ax.set_ylim(0.5, 0.72)
    ax.set_xlabel("Human-rodent protein identity (%, quartile median)")
    ax.set_ylabel("Directional agreement")
    ax.legend(loc="upper right")
    panel_label(ax, "A", "Transfer vs sequence conservation")

    # Panel B: forest of external anchors (odds ratios)
    ax = axes[1]
    style_panel(ax, grid_axis="x")
    items = [
        ("Protein identity\n(per +10 pts)", trend["or_per_10pct"],
         trend["or_lo_per_10pct"], trend["or_hi_per_10pct"]),
        ("Ensembl 1:1\northolog", orth.loc["ensembl_one2one_both", "odds_ratio"],
         orth.loc["ensembl_one2one_both", "or_lo"], orth.loc["ensembl_one2one_both", "or_hi"]),
        ("NCBI 1:1\northolog", orth.loc["ncbi_1to1", "odds_ratio"],
         orth.loc["ncbi_1to1", "or_lo"], orth.loc["ncbi_1to1", "or_hi"]),
    ]
    y = np.arange(len(items))[::-1]
    for yi, (_, orv, olo, ohi) in zip(y, items):
        ax.plot([olo, ohi], [yi, yi], color=PALETTE["ink"], lw=1.6, zorder=3)
        ax.scatter([orv], [yi], s=95, color="white", edgecolors=PALETTE["navy"],
                   linewidths=2.0, zorder=5)
        ax.text(ohi + 0.01, yi, f"{orv:.2f} ({olo:.2f}-{ohi:.2f})",
                va="center", ha="left", fontsize=9, color=PALETTE["ink"])
    ax.axvline(1.0, ls="--", color=PALETTE["muted"], lw=1.4, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([it[0] for it in items])
    ax.set_xlim(0.7, 1.35)
    ax.set_xlabel("Odds ratio for directional transfer")
    panel_label(ax, "B", "External anchors (null)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, "F6_conservation", cfg.figures)


def main():
    cfg = Config.default()
    fig_overview(cfg)
    fig_by_aspect(cfg)
    fig_depth(cfg)
    fig_decomposition(cfg)
    fig_prediction(cfg)
    fig_conservation(cfg)
    print("Wrote figures (png/pdf/tif) to", cfg.figures)


if __name__ == "__main__":
    main()
