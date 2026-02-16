#[cfg(test)]
mod tests {
    use crate::{VectorIndex, Quantization};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use frankensearch_core::SearchError;

    fn temp_index_path(name: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "frankensearch-index-repro-{name}-{}-{now}.fsvi",
            std::process::id()
        ))
    }

    #[test]
    fn soft_delete_rolls_back_main_index_on_wal_failure() {
        // This test simulates a WAL failure by making the WAL file read-only *after* the index is opened but *before* soft_delete is called.
        // Note: This simulation might be OS-dependent. A more robust way would be to inject a fault into the WAL writer, but that requires internal mocking.
        // For now, we'll try a filesystem-level trigger.

        let path = temp_index_path("soft-delete-rollback");
        let mut writer = VectorIndex::create_with_revision(&path, "hash", "test", 4, Quantization::F16)
            .expect("writer");
        writer.write_record("doc-a", &[1.0, 0.0, 0.0, 0.0]).expect("write doc-a");
        writer.finish().expect("finish");

        let mut index = VectorIndex::open(&path).expect("open");
        
        // Sanity check: doc-a is live
        let idx = index.find_index_by_doc_id("doc-a").expect("find").expect("some");
        assert!(!index.is_deleted(idx));

        // Locate the WAL file
        let wal_path = crate::wal::wal_path_for(&path);
        
        // Create a dummy WAL file so we can mess with its permissions/state
        // Wait, normally the WAL is created on first append/delete.
        // Let's force a WAL creation by appending something first.
        index.append("doc-b", &[0.0, 1.0, 0.0, 0.0]).expect("append doc-b");
        assert!(wal_path.exists());

        // Now, make the WAL file read-only to force a write error during soft_delete_wal_entry.
        let mut perms = fs::metadata(&wal_path).expect("wal metadata").permissions();
        perms.set_readonly(true);
        fs::set_permissions(&wal_path, perms).expect("set readonly");

        // Attempt soft_delete. It should fail due to WAL write error.
        let result = index.soft_delete("doc-a");
        
        // Restore permissions so we can clean up
        let mut perms = fs::metadata(&wal_path).expect("wal metadata").permissions();
        perms.set_readonly(false);
        fs::set_permissions(&wal_path, perms).expect("set readonly false");

        assert!(result.is_err(), "soft_delete should fail when WAL is unwritable");

        // CRITICAL CHECK: The main index tombstone should have been rolled back.
        let idx_after = index.find_index_by_doc_id("doc-a").expect("find").expect("some");
        assert!(!index.is_deleted(idx_after), "doc-a should NOT be deleted after failed soft_delete");

        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&wal_path);
    }
}
