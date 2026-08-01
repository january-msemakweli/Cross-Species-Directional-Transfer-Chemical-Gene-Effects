# Cross-species directional transfer of chemical-gene effects

Analysis code and non-identifiable aggregate outputs for an independence-guarded
meta-analysis of the Comparative Toxicogenomics Database (CTD). The study asks:
when a chemical is curated to increase or decrease a gene in rodents, how often
does the same directional call hold in humans when the two species are supported
by **disjoint** publications?

Manuscript (submitted): *Directional transfer of chemical-gene effects from
rodents to humans: an independence-guarded meta-analysis of the Comparative
Toxicogenomics Database*.

## Key results (independence-guarded primary set)

| Finding | Estimate |
|---|---|
| Human-rodent directional agreement | 62.0% (95% CI 61.5-62.5; *n* = 40,779) |
| Shared-publication (naive) agreement | 89.2% |
| Mouse-rat within-rodent benchmark | 66.2% |
| Human-specific divergence vs that benchmark | 4.2 points (95% CI −1.1 to 10.9) |
| Transferability for unseen chemicals (grouped CV AUC) | 0.58 |
| Protein % identity association with transfer | OR 0.97 per +10 points (95% CI 0.94-1.00) |

Agreement varies strongly by molecular endpoint and replication depth, and
barely by chemical class. Sequence conservation is not associated with
directional transfer. Estimates are literature-level directional concordance
conditional on curation, not controlled biological replication.

## Data availability

This repository redistributes **no raw CTD bulk files**. CTD is openly available
from [ctdbase.org/downloads](https://ctdbase.org/). NCBI Gene orthologs and
Ensembl Compara are openly available from their providers. What is included here:

- Analysis code for Aims 0-4 and figure generation
- Derived effect-key table (`derived/effect_keys.parquet`) sufficient to
  regenerate all paper tables and figures without re-parsing the CTD dump
- Slim external conservation tables under `external_data/`
- Aggregate result tables (`tables/`) and publication figures (`figures/`)
- Manuscript LaTeX sources (`latex_submission/`)

To rebuild the derived table from scratch, place the CTD bulk files so that
`CTD_chem_gene_ixns.tsv.gz`, `CTD_chemicals.tsv.gz`,
`CTD_genes_pathways.tsv.gz`, and `CTD_chem_gene_ixn_types.tsv` are visible
either in `./data/` or via the environment variable `CTD_DATA_DIR`, then run
with `--rebuild`.

## Repository layout

```
analysis_core.py          Shared config, IO, Wilson / cluster-bootstrap helpers
build_dataset.py          Aim 0: CTD interactions -> derived/effect_keys.parquet
aim1_reproducibility.py   Aim 1: independence-guarded agreement + strata
aim2_decomposition.py     Aim 2: within-species inconsistency, ICC, conflicts
aim3_prediction.py        Aim 3: grouped-CV transferability models
aim4_conservation.py      Aim 4: Ensembl / NCBI conservation anchor
make_figures.py           Figures F1-F6
figstyle.py               Shared plotting style
run_all.py                End-to-end orchestrator
analysis.ipynb            Narrative walkthrough
derived/                  Effect-key and modeling frames (parquet)
tables/                   Aggregate CSV outputs (A1-A4)
figures/                  Publication figures (PDF, PNG, TIFF)
external_data/            Orthology / % identity anchors + rebuild scripts
latex_submission/         Manuscript, supplement, highlights, title page, cover letter
requirements.txt          Python dependencies
CITATION.cff              Citation metadata
```

## Reproducing the analysis

1. Create an environment:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run from the repository root (uses the committed derived table):

```bash
python run_all.py
```

Force a rebuild from a local CTD download:

```bash
# optional: point at your CTD bulk folder
set CTD_DATA_DIR=C:\path\to\ctd\data          # Windows cmd
# export CTD_DATA_DIR=/path/to/ctd/data       # bash
python run_all.py --rebuild
```

3. Optional narrative notebook:

```bash
jupyter notebook analysis.ipynb
```

Tables are written to `tables/`; figures to `figures/`. Random seed is fixed at
`20260731` in `analysis_core.Config`.

## Method notes (brief)

- **Unit of analysis:** effect key = (ChemicalID, GeneID, molecular aspect),
  with per-species consensus direction and supporting PubMed IDs.
- **Independence guard (primary):** human and rodent PubMed ID sets are
  disjoint, so agreement is not inflated by co-curation of the same paper into
  both species.
- **Inference:** Wilson intervals plus a chemical-cluster bootstrap (1,000
  replications).
- **Within-rodent benchmark:** mouse-versus-rat agreement estimated the same
  way (blends curation reproducibility with mouse-rat biology; used as an upper
  reference, not a pure noise floor).
- **Prediction:** grouped cross-validation by chemical so performance reflects
  generalization to unseen chemicals.
- **External anchor:** Ensembl Compara protein % identity and 1:1 orthology,
  plus NCBI 1:1 status, joined outside CTD.

## Citation

If you use this code or reuse these analyses, please cite the manuscript and
this repository (see `CITATION.cff`). A Zenodo DOI will be added on the GitHub
Release once the archive is minted.

## License

Code is released under the MIT License (see `LICENSE`). CTD, NCBI, and Ensembl
data remain under their respective terms of use; this repository does not claim
ownership of those resources.

## Contact

Corresponding author: January G. Msemakweli (`jmsemak1@jh.edu`).
