//! Embed a real text corpus with a Model2Vec (potion) model → a flat little-endian
//! f32 slab (`N × dim`), plus a JSON sidecar with `{n, dim}`.
//!
//! This is the real-embedding generator behind the ANN/quantization validation
//! benches: it replaces the synthetic clustered-Gaussian corpus that every prior
//! recall number rests on with genuine static embeddings of real English text
//! (authentic per-dimension anisotropy / outlier structure).
//!
//! Run (LOCAL — needs the model files on disk):
//! ```bash
//! cargo run --release -p frankensearch-embed --features model2vec \
//!   --example potion_embed_corpus -- <model_dir> <corpus.txt> <out.bin>
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use frankensearch_embed::Model2VecEmbedder;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: potion_embed_corpus <model_dir> <corpus.txt> <out.bin>");
        std::process::exit(2);
    }
    let model_dir = &args[1];
    let corpus_path = &args[2];
    let out_path = &args[3];

    let emb = Model2VecEmbedder::load(model_dir).expect("load model2vec model");

    // ── Smoke check: related pair should out-score an unrelated pair. ──
    let a = emb.embed_sync("the cat sat on the warm windowsill in the sun").unwrap();
    let b = emb
        .embed_sync("a kitten rested on the sunny window ledge")
        .unwrap();
    let c = emb
        .embed_sync("quarterly revenue exceeded analyst expectations")
        .unwrap();
    eprintln!(
        "[smoke] dim={} cos(related)={:.3} cos(unrelated)={:.3}",
        a.len(),
        cosine(&a, &b),
        cosine(&a, &c)
    );

    let reader = BufReader::new(File::open(corpus_path).expect("open corpus"));
    let mut w = BufWriter::new(File::create(out_path).expect("create out"));
    let mut n = 0usize;
    let mut dim = 0usize;
    for line in reader.lines() {
        let line = line.expect("read line");
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let v = emb.embed_sync(t).expect("embed");
        // Skip all-zero (fully OOV) rows so the slab has no degenerate vectors.
        if v.iter().all(|x| *x == 0.0) {
            continue;
        }
        if dim == 0 {
            dim = v.len();
        }
        for x in &v {
            w.write_all(&x.to_le_bytes()).unwrap();
        }
        n += 1;
        if n % 5000 == 0 {
            eprintln!("[embed] {n} …");
        }
    }
    w.flush().unwrap();

    let sidecar = format!("{out_path}.meta.json");
    std::fs::write(&sidecar, format!("{{\"n\":{n},\"dim\":{dim}}}\n")).unwrap();
    eprintln!("[done] wrote {n} vectors × {dim} dim -> {out_path} (+ {sidecar})");
}
