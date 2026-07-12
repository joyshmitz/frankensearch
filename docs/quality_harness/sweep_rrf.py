#!/usr/bin/env python
"""Quality experiment: sweep RRF tier-weights and pool size on scifact.

Baseline harness uses symmetric RRF over top-100 pools (hybrid nDCG@10 = 0.6695). The
memory documents two levers: (1) up-weight the stronger single tier, (2) larger candidate
pools. Embed the corpus ONCE, compute per-query dense/lexical rankings once, then sweep
weights and pool sizes in-memory (cheap). Reports nDCG@10 vs the symmetric baseline.
"""
import json, os, re, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "hf"))
from model2vec import StaticModel
from rank_bm25 import BM25Okapi

TOKEN = re.compile(r"[a-z0-9]+")
def tok(s): return TOKEN.findall(s.lower())

def load(ds):
    d = os.path.join(HERE, ds)
    docs, doc_ids = [], []
    for line in open(os.path.join(d, "corpus.jsonl")):
        o = json.loads(line); doc_ids.append(o["_id"])
        docs.append((o.get("title", "") + " " + o.get("text", "")).strip())
    queries = {}
    for line in open(os.path.join(d, "queries.jsonl")):
        o = json.loads(line); queries[o["_id"]] = o["text"]
    qrels = {}
    with open(os.path.join(d, "qrels", "test.tsv")) as f:
        next(f)
        for line in f:
            q, dcid, rel = line.rstrip("\n").split("\t")
            if int(rel) > 0:
                qrels.setdefault(q, {})[dcid] = int(rel)
    q_ids = [q for q in queries if q in qrels]
    return doc_ids, docs, queries, q_ids, qrels

def ndcg_at_k(ranked_ids, rel, k=10):
    dcg = sum((2**rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked_ids[:k]) if rel.get(d, 0))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

def main():
    ds = "scifact"
    model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
    doc_ids, docs, queries, q_ids, qrels = load(ds)
    doc_emb = model.encode(docs, show_progress_bar=False)
    doc_emb /= (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-9)
    bm25 = BM25Okapi([tok(d) for d in docs])
    idx2id = {i: d for i, d in enumerate(doc_ids)}

    # Precompute per-query full rankings for both tiers (argsort once each).
    per_q = []
    for q in q_ids:
        qe = model.encode([queries[q]], show_progress_bar=False)[0]
        qe /= (np.linalg.norm(qe) + 1e-9)
        cos = doc_emb @ qe
        bm = np.asarray(bm25.get_scores(tok(queries[q])))
        per_q.append((np.argsort(-cos), np.argsort(-bm), qrels[q]))

    K = 60
    def eval_weighted(pool, w_dense, w_lex):
        total = 0.0
        for dense_order, lex_order, rel in per_q:
            dtop, ltop = dense_order[:pool], lex_order[:pool]
            drank = {idx2id[i]: r for r, i in enumerate(dtop)}
            lrank = {idx2id[i]: r for r, i in enumerate(ltop)}
            ids = set(drank) | set(lrank)
            BIG = 10**9
            fused = {d: w_dense / (K + drank.get(d, BIG)) + w_lex / (K + lrank.get(d, BIG)) for d in ids}
            ranked = sorted(fused, key=lambda d: -fused[d])
            total += ndcg_at_k(ranked, rel)
        return total / len(per_q)

    base = eval_weighted(100, 1.0, 1.0)
    print(f"[baseline] pool=100 w=(1,1) symmetric RRF nDCG@10 = {base:.4f}")
    print("[sweep] (pool, w_dense, w_lex) -> nDCG@10  (delta vs baseline)")
    best = (base, (100, 1.0, 1.0))
    for pool in (50, 100, 200):
        for (wd, wl) in [(1,1), (0.75,1.25), (0.5,1.5), (0.6,1.4), (1.25,0.75), (0.4,1.6)]:
            score = eval_weighted(pool, wd, wl)
            flag = "  <== best" if score > best[0] else ""
            if score > best[0]:
                best = (score, (pool, wd, wl))
            print(f"  pool={pool:>3} w=({wd},{wl})  {score:.4f}  ({score-base:+.4f}){flag}")
    print(f"[best] {best[1]} nDCG@10={best[0]:.4f}  gain_vs_symmetric={best[0]-base:+.4f}")

if __name__ == "__main__":
    main()
