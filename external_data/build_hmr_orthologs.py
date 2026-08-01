"""
Build a slim human-mouse-rat ortholog table from NCBI gene_orthologs.gz
for Aim 4 of the cross-species directional-transfer study.

Primary organism rows in gene_orthologs are typically human (tax 9606).
We keep only pairs where Other_tax_id is mouse (10090) or rat (10116),
then collapse to one row per human GeneID with mouse/rat GeneIDs and
simple conservation features:
  - has_mouse / has_rat / has_both
  - n_mouse / n_rat (counts within the ortholog set; >1 implies non-1:1)
  - is_1to1_hmr (exactly one mouse and one rat partner)
"""
from __future__ import annotations

import gzip
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
HUMAN, MOUSE, RAT = 9606, 10090, 10116

# gene_orthologs: #tax_id GeneID relationship Other_tax_id Other_GeneID
mouse_of = defaultdict(set)
rat_of = defaultdict(set)
n_rows = 0
with gzip.open(ROOT / "gene_orthologs.gz", "rt", encoding="utf-8", errors="replace") as fh:
    header = fh.readline()
    for line in fh:
        n_rows += 1
        tax, gid, rel, otax, ogid = line.rstrip("\n").split("\t")
        if int(tax) != HUMAN:
            continue
        otax = int(otax)
        if otax == MOUSE:
            mouse_of[gid].add(ogid)
        elif otax == RAT:
            rat_of[gid].add(ogid)
        if n_rows % 2_000_000 == 0:
            print(f"  ...scanned {n_rows:,} ortholog rows")

human_ids = sorted(set(mouse_of) | set(rat_of), key=lambda x: int(x))
rows = []
for hid in human_ids:
    m = sorted(mouse_of.get(hid, set()), key=lambda x: int(x))
    r = sorted(rat_of.get(hid, set()), key=lambda x: int(x))
    rows.append(
        {
            "human_geneid": hid,
            "mouse_geneids": "|".join(m),
            "rat_geneids": "|".join(r),
            "n_mouse": len(m),
            "n_rat": len(r),
            "has_mouse": int(len(m) > 0),
            "has_rat": int(len(r) > 0),
            "has_both_rodents": int(len(m) > 0 and len(r) > 0),
            "is_1to1_hmr": int(len(m) == 1 and len(r) == 1),
        }
    )

df = pd.DataFrame(rows)
out = ROOT / "human_mouse_rat_orthologs.parquet"
df.to_parquet(out, index=False)
df.to_csv(ROOT / "human_mouse_rat_orthologs.tsv.gz", sep="\t", index=False, compression="gzip")

print(f"Scanned ortholog rows: {n_rows:,}")
print(f"Human genes with mouse and/or rat ortholog: {len(df):,}")
print(f"  with mouse: {df['has_mouse'].sum():,}")
print(f"  with rat:   {df['has_rat'].sum():,}")
print(f"  with both:  {df['has_both_rodents'].sum():,}")
print(f"  1:1 H-M-R:  {df['is_1to1_hmr'].sum():,}")
print(f"Wrote {out.name} and human_mouse_rat_orthologs.tsv.gz")


# Also build a HomoloGene (build68) legacy map: human GeneID -> group size + mouse/rat IDs
# homologene.data columns (no header): HID tax_id GeneID Symbol ProteinGI ProteinAccession
hg_groups = defaultdict(list)
with open(ROOT / "homologene.data", "r", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        hid, tax, gid = parts[0], int(parts[1]), parts[2]
        if tax in (HUMAN, MOUSE, RAT):
            hg_groups[hid].append((tax, gid, parts[3] if len(parts) > 3 else ""))

hg_rows = []
for hid, members in hg_groups.items():
    humans = [g for t, g, s in members if t == HUMAN]
    mice = [g for t, g, s in members if t == MOUSE]
    rats = [g for t, g, s in members if t == RAT]
    if not humans:
        continue
    # one row per human gene in the group
    for hgid in humans:
        hg_rows.append(
            {
                "homologene_id": hid,
                "human_geneid": hgid,
                "mouse_geneids": "|".join(mice),
                "rat_geneids": "|".join(rats),
                "n_mouse": len(mice),
                "n_rat": len(rats),
                "group_n_hmr": len(humans) + len(mice) + len(rats),
                "has_both_rodents": int(len(mice) > 0 and len(rats) > 0),
                "is_1to1_hmr": int(len(humans) == 1 and len(mice) == 1 and len(rats) == 1),
            }
        )

hg = pd.DataFrame(hg_rows)
hg.to_parquet(ROOT / "homologene68_human_mouse_rat.parquet", index=False)
hg.to_csv(ROOT / "homologene68_human_mouse_rat.tsv.gz", sep="\t", index=False, compression="gzip")
print(f"\nHomoloGene68 human genes mapped: {len(hg):,}")
print(f"  with both rodents: {hg['has_both_rodents'].sum():,}")
print(f"  1:1 H-M-R: {hg['is_1to1_hmr'].sum():,}")
print("Wrote homologene68_human_mouse_rat.parquet / .tsv.gz")
