"""
Shared core for the cross-species directional-transfer study.

See PROPOSAL.md. Everything downstream keys off an "effect key":
    (ChemicalID, GeneID, aspect)
with, per species, the set of curated directions and supporting PubMed IDs.
"""
from __future__ import annotations

import gzip
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HUMAN = "9606"
MOUSE = "10090"
RAT = "10116"
RODENTS = (MOUSE, RAT)

# Molecular aspects retained. These are the single-action "increases/decreases^X"
# effects that carry an interpretable direction.
DIRECTIONAL_ASPECTS = {
    "expression",
    "methylation",
    "activity",
    "phosphorylation",
    "secretion",
    "stability",
    "localization",
    "splicing",
    "metabolic processing",
    "abundance",
    "uptake",
    "transport",
    "export",
    "acetylation",
    "ubiquitination",
    "cleavage",
    "response to substance",
}

DIRECTIONS = ("increases", "decreases")


@dataclass
class Config:
    root: Path
    data_dir: Path
    derived: Path
    tables: Path
    figures: Path
    seed: int = 20260731
    min_stratum: int = 30
    n_boot: int = 1000

    @classmethod
    def default(cls) -> "Config":
        root = Path(__file__).resolve().parent
        # Prefer CTD_DATA_DIR, then ./data (public-repo layout), then ../data
        # (local sibling download used during development).
        if os.environ.get("CTD_DATA_DIR"):
            data_dir = Path(os.environ["CTD_DATA_DIR"]).expanduser().resolve()
        else:
            candidates = (root / "data", root.parent / "data")
            data_dir = next(
                (p for p in candidates if (p / "CTD_chem_gene_ixns.tsv.gz").exists()),
                root / "data",
            )
        cfg = cls(
            root=root,
            data_dir=data_dir,
            derived=root / "derived",
            tables=root / "tables",
            figures=root / "figures",
        )
        for d in (cfg.derived, cfg.tables, cfg.figures):
            d.mkdir(exist_ok=True)
        return cfg


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _fields_and_stream(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    fields = None
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if fields is None:
                if line.startswith("# Fields:"):
                    for l2 in fh:
                        if l2.startswith("#") and "\t" in l2:
                            fields = l2.lstrip("#").strip().split("\t")
                            break
                continue
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            n = len(fields)
            if len(cols) > n:
                cols = cols[:n]
            elif len(cols) < n:
                cols = cols + [""] * (n - len(cols))
            yield dict(zip(fields, cols))


def read_ctd_tsv(path: Path) -> pd.DataFrame:
    """Full read via manual tab split (CTD free-text breaks pandas quoting)."""
    rows = list(_fields_and_stream(path))
    return pd.DataFrame(rows).replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA})


def iter_chem_gene(path: Path):
    """Yield parsed single-action directional chem-gene rows only."""
    for row in _fields_and_stream(path):
        ia = row.get("InteractionActions", "")
        if not ia or "|" in ia or "^" not in ia:
            continue
        direction, aspect = ia.split("^", 1)
        if direction not in DIRECTIONS or aspect not in DIRECTIONAL_ASPECTS:
            continue
        yield row, direction, aspect


def bare_id(x) -> str | float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return s.split(":", 1)[-1] if ":" in s else s


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson_ci(successes: int, n: int, alpha: float = 0.05):
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
):
    """Cluster (block) bootstrap of a mean; clusters resampled with replacement.

    values: 0/1 array. clusters: cluster id per obs (e.g. chemical).
    Returns (point, lo, hi).
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    point = float(values.mean()) if values.size else np.nan
    # index observations by cluster
    order = np.argsort(clusters, kind="stable")
    sv = values[order]
    sc = clusters[order]
    uniq, starts = np.unique(sc, return_index=True)
    ends = np.append(starts[1:], len(sc))
    groups = [sv[s:e] for s, e in zip(starts, ends)]
    n_groups = len(groups)
    if n_groups < 2:
        return point, np.nan, np.nan
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        cat = np.concatenate([groups[i] for i in pick])
        boots[b] = cat.mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return point, float(lo), float(hi)


def two_by_two(a: int, b: int, c: int, d: int) -> dict:
    table = np.array([[a, b], [c, d]], dtype=int)
    oddsratio, fisher_p = stats.fisher_exact(table)
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    if min(a, b, c, d) == 0:
        oddsratio = (aa * dd) / (bb * cc)
    log_or = math.log(oddsratio) if oddsratio > 0 else np.nan
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    z = stats.norm.ppf(0.975)
    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "or": float(oddsratio),
        "or_lo": float(math.exp(log_or - z * se)),
        "or_hi": float(math.exp(log_or + z * se)),
        "fisher_p": float(fisher_p),
    }
