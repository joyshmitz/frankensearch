use frankensearch_core::SearchResult;
use std::path::PathBuf;
use std::fs;

use crate::{VectorIndex, TwoTierConfig, Quantization};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_index_path(name: &str) -> PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("test-wal-shadow-{name}-{now}"))
}

#[test]
fn stale_main_entry_leaks() {
    let path = temp_index_path("stale-leak");
    
    // Create main index with [1.0, 0.0]
    let mut writer = VectorIndex::create_with_revision(&path, "test", "r1", 2, Quantization::F32).unwrap();
    writer.write_record("doc-a", &[1.0, 0.0]).unwrap();
    writer.finish().unwrap();

    let mut index = VectorIndex::open(&path).unwrap();
    
    // Append doc-a with [0.0, 1.0] to WAL
    index.append("doc-a", &[0.0, 1.0]).unwrap();
    
    // Search for [1.0, 0.0]. The WAL entry scores 0.0, the Main entry scores 1.0.
    // If the bug exists, the Main entry will be returned.
    let hits = index.search_top_k(&[1.0, 0.0], 1, None).unwrap();
    
    // Wait, the new WAL entry is [0.0, 1.0]. It shouldn't match [1.0, 0.0] well.
    // But does the search return the OLD doc-a?
    // Let's assert it DOES NOT return doc-a.
    // Actually, if K=1, the only document is doc-a. 
    // The WAL entry scores 0.0. The Main entry scores 1.0.
    // If Main leaks, hits[0].score == 1.0.
    // If it's correctly shadowed, the only hit should be doc-a with score 0.0.
    assert_eq!(hits[0].score, 0.0, "Expected score 0.0 from WAL entry, but got leaked score {}", hits[0].score);
}
