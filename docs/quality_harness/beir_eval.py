#!/usr/bin/env python
"""Rebuilt pure-Python BEIR hybrid eval (model2vec potion + rank_bm25 BM25Okapi).

Reproduces the campaign's quality harness: per corpus, score every doc by the dense
tier (cosine of potion embeddings) and the lexical tier (BM25Okapi), fuse by RRF, and
report nDCG@10 for each tier and the hybrid. The campaign's headline invariant is
"hybrid >= best single tier". Basic whitespace+lowercase tokenization for the baseline
(stem+stop refinement is a documented follow-up).
"""
import json, os, re, sys, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "hf"))
from model2vec import StaticModel
from rank_bm25 import BM25Okapi

RRF_K = 60
TOKEN = re.compile(r"[a-z0-9]+")

def tok(s): return TOKEN.findall(s.lower())

def load(ds):
    d = os.path.join(HERE, ds)
    docs, doc_ids = [], []
    for line in open(os.path.join(d, "corpus.jsonl")):
        o = json.loads(line)
        doc_ids.append(o["_id"])
        docs.append((o.get("title", "") + " " + o.get("text", "")).strip())
    queries, q_ids = {}, []
    for line in open(os.path.join(d, "queries.jsonl")):
        o = json.loads(line)
        queries[o["_id"]] = o["text"]
    qrels = {}
    with open(os.path.join(d, "qrels", "test.tsv")) as f:
        next(f)  # header
        for line in f:
            q, dcid, rel = line.rstrip("\n").split("\t")
            if int(rel) > 0:
                qrels.setdefault(q, {})[dcid] = int(rel)
    # only queries that have qrels in the test split
    q_ids = [q for q in queries if q in qrels]
    return doc_ids, docs, queries, q_ids, qrels

def ndcg_at_k(ranked_ids, rel, k=10):
    dcg = 0.0
    for i, did in enumerate(ranked_ids[:k]):
        g = rel.get(did, 0)
        if g:
            dcg += (2**g - 1) / math.log2(i + 2)
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

def rrf(rank_map_a, rank_map_b, k=RRF_K):
    ids = set(rank_map_a) | set(rank_map_b)
    return {d: 1.0/(k+rank_map_a.get(d, 10**9)) + 1.0/(k+rank_map_b.get(d, 10**9)) for d in ids}

def eval_ds(ds, model):
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    doc_emb = model.encode(docs, show_progress_bar=False)
    doc_emb /= (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-9)
    tokenized = [tok(d) for d in docs]
    bm25 = BM25Okapi(tokenized)
    id_index = {i: did for i, did in enumerate(doc_ids)}

    n_dense = n_lex = n_hyb = 0.0
    for q in q_ids:
        qt = queries[q]
        qe = model.encode([qt], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = doc_emb @ qe
        bm = bm25.get_scores(tok(qt))
        # top-100 pools per tier for the RRF fusion (engine-faithful: fusion over pools)
        dense_top = np.argsort(-cos)[:100]
        lex_top = np.argsort(-bm)[:100]
        dense_rank = {id_index[i]: r for r, i in enumerate(dense_top)}
        lex_rank = {id_index[i]: r for r, i in enumerate(lex_top)}
        rel = qrels[q]
        dense_ranked = [id_index[i] for i in dense_top]
        lex_ranked = [id_index[i] for i in lex_top]
        fused = rrf(dense_rank, lex_rank)
        hyb_ranked = [d for d, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
        n_dense += ndcg_at_k(dense_ranked, rel)
        n_lex += ndcg_at_k(lex_ranked, rel)
        n_hyb += ndcg_at_k(hyb_ranked, rel)
    n = len(q_ids)
    return n, n_dense/n, n_lex/n, n_hyb/n

def main():
    datasets = sys.argv[1:] or ["scifact"]
    print("[harness] loading potion-retrieval-32M (model2vec StaticModel)...", flush=True)
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    print(f"[harness] model loaded, dim={model.dim}", flush=True)
    for ds in datasets:
        n, dense, lex, hyb = eval_ds(ds, model)
        best = max(dense, lex)
        print(f"[result] ds={ds} q={n} nDCG@10 dense={dense:.4f} lexical={lex:.4f} "
              f"hybrid={hyb:.4f} best_single={best:.4f} hybrid_gain_vs_best={hyb-best:+.4f} "
              f"hybrid_ge_best={hyb>=best-1e-6}", flush=True)

if __name__ == "__main__":
    main()
