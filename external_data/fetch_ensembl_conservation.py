"""
Fetch graded human->rodent protein conservation (% identity) from Ensembl
BioMart, keyed by NCBI Entrez GeneID (= CTD GeneID), for Aim 4.

BioMart forbids mixing attributes from different attribute pages in one query
(entrezgene_id is on the feature page, homolog %id is on the homolog page), so
we run three queries and join on ensembl_gene_id:
  1. ensembl_gene_id -> entrezgene_id
  2. ensembl_gene_id -> mouse homolog %id + orthology type
  3. ensembl_gene_id -> rat   homolog %id + orthology type

Output: ensembl_conservation_hmr.parquet / .tsv.gz  (keyed by NCBI GeneID)
"""
from __future__ import annotations

import io
import time
import urllib.parse as _up
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

MIRRORS = [
    "https://www.ensembl.org/biomart/martservice",
    "https://asia.ensembl.org/biomart/martservice",
]


def build_query(attributes: list[str]) -> str:
    attrs = "".join(f'<Attribute name="{a}"/>' for a in attributes)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<!DOCTYPE Query>'
        '<Query virtualSchemaName="default" formatter="TSV" header="0" '
        'uniqueRows="1" count="" datasetConfigVersion="0.6">'
        '<Dataset name="hsapiens_gene_ensembl" interface="default">'
        f'{attrs}'
        '</Dataset>'
        '</Query>'
    )


def fetch(attributes: list[str], colnames: list[str], label: str) -> pd.DataFrame:
    xml = build_query(attributes)
    body = ("query=" + _up.quote(xml)).encode("utf-8")
    last = None
    for mirror in MIRRORS:
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    mirror, data=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded",
                             "User-Agent": "Mozilla/5.0 (ctd-aim4 research)"},
                )
                with urllib.request.urlopen(req, timeout=240) as r:
                    txt = r.read().decode("utf-8", errors="replace")
                if "ERROR" in txt[:300] or "Exception" in txt[:300]:
                    raise RuntimeError(txt[:300])
                df = pd.read_csv(io.StringIO(txt), sep="\t", header=None, names=colnames,
                                 dtype=str)
                print(f"  {label}: {mirror.split('//')[1].split('/')[0]} -> rows={len(df):,}")
                if len(df):
                    return df
            except Exception as e:
                last = e
                print(f"  {label}: {mirror.split('//')[1].split('/')[0]} "
                      f"attempt {attempt+1} failed: {str(e)[:120]}")
                time.sleep(3)
    raise RuntimeError(f"All mirrors failed for {label}: {last}")


def main():
    print("1/3 Ensembl gene -> Entrez map...")
    emap = fetch(["ensembl_gene_id", "entrezgene_id"],
                 ["ensembl_gene_id", "gene"], "entrez")
    emap = emap.dropna(subset=["gene"])
    emap["gene"] = emap["gene"].astype(float).astype("Int64").astype(str)

    print("2/3 Mouse homology (%id)...")
    mouse = fetch(
        ["ensembl_gene_id", "mmusculus_homolog_orthology_type",
         "mmusculus_homolog_perc_id"],
        ["ensembl_gene_id", "otype_mouse", "perc_id_mouse"], "mouse")

    print("3/3 Rat homology (%id)...")
    rat = fetch(
        ["ensembl_gene_id", "rnorvegicus_homolog_orthology_type",
         "rnorvegicus_homolog_perc_id"],
        ["ensembl_gene_id", "otype_rat", "perc_id_rat"], "rat")

    # save raw
    emap.to_csv(ROOT / "biomart_entrez_map.tsv.gz", sep="\t", index=False, compression="gzip")
    mouse.to_csv(ROOT / "biomart_mouse_raw.tsv.gz", sep="\t", index=False, compression="gzip")
    rat.to_csv(ROOT / "biomart_rat_raw.tsv.gz", sep="\t", index=False, compression="gzip")

    def reduce_homolog(df, sp):
        df = df.copy()
        df[f"perc_id_{sp}"] = pd.to_numeric(df[f"perc_id_{sp}"], errors="coerce")
        # keep best (highest %id) homolog per ensembl gene
        df = df.sort_values(f"perc_id_{sp}", ascending=False).drop_duplicates("ensembl_gene_id")
        return df[["ensembl_gene_id", f"otype_{sp}", f"perc_id_{sp}"]]

    m = reduce_homolog(mouse, "mouse")
    r = reduce_homolog(rat, "rat")

    cons = (emap.drop_duplicates("ensembl_gene_id")
            .merge(m, on="ensembl_gene_id", how="left")
            .merge(r, on="ensembl_gene_id", how="left"))
    # collapse to one row per NCBI gene: take best identity across ensembl ids
    cons["perc_id_mouse"] = pd.to_numeric(cons["perc_id_mouse"], errors="coerce")
    cons["perc_id_rat"] = pd.to_numeric(cons["perc_id_rat"], errors="coerce")
    cons = cons.sort_values(["gene", "perc_id_mouse", "perc_id_rat"],
                            ascending=[True, False, False]).drop_duplicates("gene")

    cons["perc_id_rodent_mean"] = cons[["perc_id_mouse", "perc_id_rat"]].mean(axis=1)
    cons["perc_id_rodent_min"] = cons[["perc_id_mouse", "perc_id_rat"]].min(axis=1)
    cons["is_1to1_mouse"] = cons["otype_mouse"].eq("ortholog_one2one")
    cons["is_1to1_rat"] = cons["otype_rat"].eq("ortholog_one2one")

    keep = ["gene", "ensembl_gene_id", "perc_id_mouse", "perc_id_rat",
            "perc_id_rodent_mean", "perc_id_rodent_min",
            "otype_mouse", "otype_rat", "is_1to1_mouse", "is_1to1_rat"]
    cons = cons[keep]
    cons.to_parquet(ROOT / "ensembl_conservation_hmr.parquet", index=False)
    cons.to_csv(ROOT / "ensembl_conservation_hmr.tsv.gz", sep="\t", index=False, compression="gzip")

    print(f"\nConservation table: {len(cons):,} NCBI genes")
    n_mouse = cons["perc_id_mouse"].notna().sum()
    n_rat = cons["perc_id_rat"].notna().sum()
    print(f"  with mouse %id: {n_mouse:,}   with rat %id: {n_rat:,}")
    print("\nperc_id_rodent_mean describe:")
    print(cons["perc_id_rodent_mean"].describe().to_string())
    print("\nhead:")
    print(cons.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
