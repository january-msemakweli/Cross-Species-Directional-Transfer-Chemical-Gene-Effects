"""
Aim 0: build the analytic effect-key table.

For every (ChemicalID, GeneID, aspect) we record, per species group (human /
mouse / rat), the curated direction votes and the supporting PubMed IDs. We then
derive:
  - per-species consensus direction (+1 increases, -1 decreases, 0 ambiguous)
  - per-species vote counts and n distinct PMIDs
  - cross-species PMID overlap (independence guard)
  - agreement / conflict labels (human vs pooled rodent, and mouse vs rat)
Feature joins (chemical class, gene pathway degree, tissue breadth) are added.

Output: derived/effect_keys.parquet  (one row per chem x gene x aspect that is
seen in >=1 of the three species).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_core import (
    Config,
    HUMAN,
    MOUSE,
    RAT,
    bare_id,
    iter_chem_gene,
    read_ctd_tsv,
)

SPECIES_MAP = {HUMAN: "human", MOUSE: "mouse", RAT: "rat"}


def consensus(inc: int, dec: int) -> int:
    if inc > 0 and dec == 0:
        return 1
    if dec > 0 and inc == 0:
        return -1
    return 0  # ambiguous (both) or none


def build_keys(cfg: Config) -> pd.DataFrame:
    path = cfg.data_dir / "CTD_chem_gene_ixns.tsv.gz"
    # store[(chem,gene,aspect)][species] = [inc, dec, pmidset]
    store: dict = defaultdict(
        lambda: {
            "human": [0, 0, set()],
            "mouse": [0, 0, set()],
            "rat": [0, 0, set()],
        }
    )
    n = 0
    for row, direction, aspect in iter_chem_gene(path):
        org = row.get("OrganismID")
        sp = SPECIES_MAP.get(org)
        if sp is None:
            continue
        key = (row["ChemicalID"], row["GeneID"], aspect)
        cell = store[key][sp]
        if direction == "increases":
            cell[0] += 1
        else:
            cell[1] += 1
        pm = row.get("PubMedIDs") or ""
        for p in pm.split("|"):
            if p.strip():
                cell[2].add(p.strip())
        n += 1
        if n % 300000 == 0:
            print(f"  ...parsed {n:,} directional rows; keys so far {len(store):,}")
    print(f"Total directional rows parsed: {n:,}; unique keys: {len(store):,}")

    recs = []
    for (chem, gene, aspect), sp in store.items():
        h_inc, h_dec, h_pm = sp["human"]
        m_inc, m_dec, m_pm = sp["mouse"]
        r_inc, r_dec, r_pm = sp["rat"]
        rod_inc = m_inc + r_inc
        rod_dec = m_dec + r_dec
        rod_pm = m_pm | r_pm
        h_cons = consensus(h_inc, h_dec)
        rod_cons = consensus(rod_inc, rod_dec)
        m_cons = consensus(m_inc, m_dec)
        r_cons = consensus(r_inc, r_dec)
        shared_hr = h_pm & rod_pm
        recs.append(
            {
                "chem": chem,
                "gene": gene,
                "aspect": aspect,
                "h_inc": h_inc,
                "h_dec": h_dec,
                "rod_inc": rod_inc,
                "rod_dec": rod_dec,
                "m_inc": m_inc,
                "m_dec": m_dec,
                "r_inc": r_inc,
                "r_dec": r_dec,
                "h_pmids": len(h_pm),
                "rod_pmids": len(rod_pm),
                "h_cons": h_cons,
                "rod_cons": rod_cons,
                "m_cons": m_cons,
                "r_cons": r_cons,
                "in_human": (h_inc + h_dec) > 0,
                "in_rodent": (rod_inc + rod_dec) > 0,
                "in_mouse": (m_inc + m_dec) > 0,
                "in_rat": (r_inc + r_dec) > 0,
                "n_shared_hr_pmids": len(shared_hr),
                "pmid_disjoint_hr": len(shared_hr) == 0,
            }
        )
    df = pd.DataFrame.from_records(recs)
    return df


def add_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # --- chemical class from MeSH tree ---
    chem = read_ctd_tsv(cfg.data_dir / "CTD_chemicals.tsv.gz")[
        ["ChemicalID", "ChemicalName", "TreeNumbers"]
    ].copy()
    chem["chem_bare"] = chem["ChemicalID"].map(bare_id)
    chem = chem.drop_duplicates("chem_bare")
    df["chem_bare"] = df["chem"].map(bare_id)
    df = df.merge(
        chem[["chem_bare", "ChemicalName", "TreeNumbers"]].rename(
            columns={"ChemicalName": "chem_name", "TreeNumbers": "chem_trees"}
        ),
        on="chem_bare",
        how="left",
    )
    df["chem_class"] = df["chem_trees"].map(_chem_class)

    # --- gene pathway degree (Reactome/KEGG membership count) ---
    gp = read_ctd_tsv(cfg.data_dir / "CTD_genes_pathways.tsv.gz")
    gp_deg = gp.groupby("GeneID")["PathwayID"].nunique().rename("gene_pathway_deg")
    df = df.merge(gp_deg, left_on="gene", right_index=True, how="left")
    df["gene_pathway_deg"] = df["gene_pathway_deg"].fillna(0).astype(int)

    # --- gene name ---
    genes = read_ctd_tsv(cfg.data_dir / "CTD_genes.tsv.gz")[["GeneID", "GeneSymbol"]]
    genes = genes.drop_duplicates("GeneID")
    df = df.merge(genes.rename(columns={"GeneSymbol": "gene_symbol"}), left_on="gene", right_on="GeneID", how="left")

    return df


# coarse chemical classes for stratification
_CLASS_TREE = [
    ("metals_inorganic", ("D01.268", "D01.552.544", "D01.248")),
    ("organic_pollutants", ("D02.455.426.559", "D02.705", "D04")),
    ("pesticides", ("D27.888.723",)),
    ("drugs_therapeutic", ("D27.505",)),
    ("solvents_industrial", ("D02.455",)),
]


def _chem_class(trees) -> str:
    if pd.isna(trees):
        return "other"
    t = str(trees)
    for name, prefixes in _CLASS_TREE:
        if any(p in t for p in prefixes):
            return name
    if t.startswith("D03") or t.startswith("D04"):
        return "heterocyclic_organic"
    return "other"


def main():
    cfg = Config.default()
    print("Building effect-key store from chem_gene_ixns...")
    df = build_keys(cfg)
    print("Adding features...")
    df = add_features(cfg=cfg, df=df)
    out = cfg.derived / "effect_keys.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out}  shape={df.shape}")

    # quick sanity snapshot
    both = df[df["in_human"] & df["in_rodent"]]
    print(f"in both species: {len(both):,}")
    print(f"pmid-disjoint of those: {both['pmid_disjoint_hr'].mean()*100:.1f}%")
    unamb = both[(both['h_cons'] != 0) & (both['rod_cons'] != 0)]
    agree = (unamb['h_cons'] == unamb['rod_cons']).mean()
    print(f"unambiguous both n={len(unamb):,} agree={agree*100:.1f}%")


if __name__ == "__main__":
    main()
