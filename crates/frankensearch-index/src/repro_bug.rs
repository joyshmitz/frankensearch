
#[cfg(test)]
mod tests {
    use crate::{Quantization, VectorIndex, wal};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use std::fs;

    fn temp_index_path(name: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-repro-{name}-{}-{now}.fsvi",
            std::process::id()
        ))
    }

    #[test]
    fn repro_duplicate_entries_on_compaction_crash() {
        let path = temp_index_path("compaction-crash");
        let dim = 4;
        
        // 1. Create initial index with 1 document
        let mut writer = VectorIndex::create_with_revision(&path, "test", "v1", dim, Quantization::F16).unwrap();
        writer.write_record("doc-A", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        writer.finish().unwrap();

        let mut index = VectorIndex::open(&path).unwrap();
        
        // 2. Append a document to WAL
        index.append("doc-B", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        
        // Check state before "compaction"
        let hits = index.search_top_k(&[1.0, 1.0, 0.0, 0.0], 10, None).unwrap();
        assert_eq!(hits.len(), 2);
        
        // 3. Simulate compaction crash:
        // We want to create a state where "doc-B" is in Main Index AND in WAL.
        // We can do this by running `compact` but preventing the WAL deletion.
        // Since we can't easily interrupt `compact`, we'll simulate the filesystem state.
        
        // Close index to flush everything
        drop(index);

        // Manually create the "post-compaction" main index that includes both A and B.
        let mut compact_writer = VectorIndex::create_with_revision(&path, "test", "v1", dim, Quantization::F16)
            .unwrap()
            .with_generation(2); // Simulate correct compaction increment
        compact_writer.write_record("doc-A", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        compact_writer.write_record("doc-B", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        compact_writer.finish().unwrap(); // Overwrites `path` with new index containing A and B.

        // Restore the WAL file (because `finish` doesn't touch it, but we need to ensure it exists and has doc-B)
        // Actually, `finish` overwrites `path`. The WAL file is at `path.wal`.
        // We didn't delete `path.wal`. So `path.wal` still contains "doc-B".
        
        // 4. Re-open index. It should load Main (A, B) and WAL (B).
        let index_reopened = VectorIndex::open(&path).unwrap();
        
        // 5. Search. If bug exists, we'll see "doc-B" twice.
        let hits = index_reopened.search_top_k(&[1.0, 1.0, 0.0, 0.0], 10, None).unwrap();
        
        // Debug output
        for hit in &hits {
            println!("Hit: {} score={}", hit.doc_id, hit.score);
        }

        // Clean up
        let _ = fs::remove_file(&path);
        let _ = wal::remove_wal(&wal::wal_path_for(&path));

        // Assert failure
        assert_eq!(hits.len(), 2, "Should have exactly 2 hits (A and B), found {}", hits.len());
        let b_count = hits.iter().filter(|h| h.doc_id == "doc-B").count();
        assert_eq!(b_count, 1, "Should have exactly 1 'doc-B', found {}", b_count);
    }
}
