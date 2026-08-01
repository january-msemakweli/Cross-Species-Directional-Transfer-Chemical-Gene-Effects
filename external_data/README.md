# External conservation anchors (Aim 4)

Processed human-mouse-rat orthology and Ensembl Compara protein-identity tables
used by `aim4_conservation.py`. These files are independent of CTD curation.

## Included (committed)

| File | Description |
|---|---|
| `ensembl_conservation_hmr.parquet` | Human Entrez GeneID with mouse/rat protein % identity and 1:1 flags |
| `human_mouse_rat_orthologs.parquet` | NCBI `gene_orthologs` collapsed to human-mouse-rat 1:1 status |
| `homologene68_human_mouse_rat.parquet` | Legacy HomoloGene (optional; not required by the primary Aim 4 path) |
| `fetch_ensembl_conservation.py` | Rebuild Ensembl table via BioMart |
| `build_hmr_orthologs.py` | Rebuild NCBI table from `gene_orthologs.gz` |

Gzipped TSV mirrors of the same tables are also included.

## Rebuild (optional)

Raw NCBI `gene_orthologs.gz` is large and not redistributed here. To rebuild:

1. Download NCBI Gene orthologs (`gene_orthologs.gz`) into this folder.
2. Run `python build_hmr_orthologs.py`.
3. Run `python fetch_ensembl_conservation.py` (requires network access to Ensembl BioMart).

The committed parquet files are sufficient to reproduce Aim 4 without those steps.
