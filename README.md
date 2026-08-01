# Cross-species directional transfer of chemical-gene effects

Code, aggregate tables, and figures for an independence-guarded meta-analysis of
the Comparative Toxicogenomics Database (CTD). When a chemical is curated to
increase or decrease a gene in rodents, how often does the same directional call
hold in humans when the two species are supported by disjoint publications?

## Contents

```
*.py                    Analysis pipeline (Aims 0-4) and figure generation
run_all.py              End-to-end orchestrator
requirements.txt        Python dependencies
tables/                 Aggregate CSV outputs used in the paper
figures/                Publication figures F1-F6 (PDF, PNG, TIFF)
```

## Reproduce the tables and figures

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` rebuilds the derived effect-key table from a local CTD bulk
download if needed, then regenerates `tables/` and `figures/`.

Point the pipeline at your CTD files with either:

- a `data/` folder in this repository containing
  `CTD_chem_gene_ixns.tsv.gz` (and the companion chemical / pathway files), or
- the environment variable `CTD_DATA_DIR` set to that folder.

CTD bulk downloads are available from [ctdbase.org/downloads](https://ctdbase.org/downloads/).
Aim 4 also expects Ensembl / NCBI orthology tables under `external_data/`
(see comments in `aim4_conservation.py`); the committed `tables/` and
`figures/` already contain the published Aim 4 outputs.

## Contact

Corresponding author: January G. Msemakweli (`jmsemak1@jh.edu`).
