//! Embed a text corpus with a RAW transformer (all-MiniLM-L6-v2 ONNX, 384-d) →
//! flat little-endian f32 slab (`N × 384`) + a `{n,dim}` sidecar.
//!
//! Unlike the Model2Vec/potion path (which PCA-smooths its embeddings to a
//! near-Gaussian per-dimension profile), raw transformer embeddings carry the
//! documented "outlier dimension" structure — the harder distribution for
//! per-dimension int8 quantization. This is the stress-test slab for
//! `real_embed_quant` / `real_embed_ann`.
//!
//! Uses `fastembed`'s synchronous `TextEmbedding::embed` directly (bypassing the
//! async Embedder wrapper). Run (LOCAL, needs the ONNX model on disk;
//! `--features fastembed` downloads the onnxruntime binary at build time):
//! ```bash
//! cargo run --release -p frankensearch-embed --features fastembed \
//!   --example minilm_embed_corpus -- <minilm_onnx_dir> <corpus.txt> <out.bin>
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use fastembed::{
    InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel,
};

fn read(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

fn flush_batch(
    model: &mut TextEmbedding,
    batch: &mut Vec<String>,
    w: &mut BufWriter<File>,
    n: &mut usize,
    dim: &mut usize,
) {
    if batch.is_empty() {
        return;
    }
    let refs: Vec<&str> = batch.iter().map(String::as_str).collect();
    let embs = model.embed(refs, None).expect("embed batch");
    for v in &embs {
        if *dim == 0 {
            *dim = v.len();
        }
        for x in v {
            w.write_all(&x.to_le_bytes()).unwrap();
        }
        *n += 1;
    }
    batch.clear();
    if *n % 5000 < 256 {
        eprintln!("[embed] {n} …");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: minilm_embed_corpus <model_dir> <corpus.txt> <out.bin>");
        std::process::exit(2);
    }
    let model_dir = Path::new(&args[1]);
    let corpus_path = &args[2];
    let out_path = &args[3];

    // Load the ONNX model + tokenizer files (fastembed UserDefinedEmbeddingModel).
    let model_bytes = read(&model_dir.join("model.onnx"));
    let tokenizer_files = TokenizerFiles {
        tokenizer_file: read(&model_dir.join("tokenizer.json")),
        config_file: read(&model_dir.join("config.json")),
        special_tokens_map_file: read(&model_dir.join("special_tokens_map.json")),
        tokenizer_config_file: read(&model_dir.join("tokenizer_config.json")),
    };
    let mut user_model = UserDefinedEmbeddingModel::new(model_bytes, tokenizer_files);
    user_model.pooling = Some(Pooling::Mean);
    let mut model =
        TextEmbedding::try_new_from_user_defined(user_model, InitOptionsUserDefined::new())
            .expect("init MiniLM");

    // Smoke check.
    let smoke = model
        .embed(
            vec![
                "the cat sat on the warm windowsill in the sun",
                "a kitten rested on the sunny window ledge",
                "quarterly revenue exceeded analyst expectations",
            ],
            None,
        )
        .expect("smoke embed");
    let cos = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    eprintln!(
        "[smoke] dim={} cos(related)={:.3} cos(unrelated)={:.3}",
        smoke[0].len(),
        cos(&smoke[0], &smoke[1]),
        cos(&smoke[0], &smoke[2])
    );

    // Stream the corpus in batches.
    let reader = BufReader::new(File::open(corpus_path).expect("open corpus"));
    let mut w = BufWriter::new(File::create(out_path).expect("create out"));
    let mut batch: Vec<String> = Vec::with_capacity(256);
    let mut n = 0usize;
    let mut dim = 0usize;

    for line in reader.lines() {
        let line = line.expect("read line");
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        batch.push(t.to_owned());
        if batch.len() == 256 {
            flush_batch(&mut model, &mut batch, &mut w, &mut n, &mut dim);
        }
    }
    flush_batch(&mut model, &mut batch, &mut w, &mut n, &mut dim);
    w.flush().unwrap();

    let sidecar = format!("{out_path}.meta.json");
    std::fs::write(&sidecar, format!("{{\"n\":{n},\"dim\":{dim}}}\n")).unwrap();
    eprintln!("[done] wrote {n} vectors × {dim} dim -> {out_path} (+ {sidecar})");
}
