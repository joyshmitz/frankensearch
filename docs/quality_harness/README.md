# Search-quality BEIR harness (pure Python)

The measured search-QUALITY vein (see `../SEARCH_QUALITY_FINDINGS.md` and the
`search-QUALITY` entries in `../NEGATIVE_EVIDENCE.md`) is evaluated with a pure-Python
harness — **no cargo** — so it lives here as durable tooling (it was repeatedly lost when
kept only in session scratchpads; rebuilt 2026-07-12).

## Setup (one-time, needs network)
```bash
python3 -m venv venv
./venv/bin/pip install model2vec rank_bm25 numpy   # model2vec is numpy-based (no torch)
# BEIR datasets (any name from the BEIR list):
for ds in scifact nfcorpus arguana scidocs; do
  curl -sL "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$ds.zip" -o "$ds.zip"
  unzip -q -o "$ds.zip"
done
```

## Run
```bash
./venv/bin/python beir_eval.py scifact nfcorpus arguana scidocs
```
Reports per corpus: nDCG@10 for the dense tier (model2vec `potion-retrieval-32M` cosine),
the lexical tier (`rank_bm25` BM25Okapi), and their RRF hybrid, plus `hybrid_ge_best`.

## Baseline (2026-07-12, scifact, basic whitespace+lowercase tokenization)
dense 0.6331 · lexical 0.6523 · hybrid 0.6695 (+0.0172 over best single) · hybrid≥best ✓ —
reproduces the campaign invariant (hybrid ≥ best single tier; lexical is the stronger tier;
dense's honest marginal value is small). Documented refinements to layer on: Tantivy-faithful
stem+stop lexical analysis, larger candidate pools, the landed pool-min-max / graph-diffusion /
query-side-hubness quality kernels.
