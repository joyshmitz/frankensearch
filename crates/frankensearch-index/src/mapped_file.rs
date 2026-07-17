//! Shared immutable file-mapping facade.
//!
//! File-backed memory maps are sound only while the mapped file's contents and
//! length remain unchanged. Frankensearch satisfies that requirement for
//! published index artifacts by writing a new temporary file, syncing it, and
//! atomically publishing it under a new generation name. Published files are
//! never modified in place; replacement and garbage collection operate on
//! directory entries instead.

use std::fs::File;
use std::io;
use std::path::Path;

use memmap2::Mmap;

/// An immutable, file-backed byte buffer.
///
/// The mapping owns its operating-system handle and remains valid after the
/// [`File`] used by [`Self::open_published`] is dropped. It exposes no mutable access.
/// Frankensearch callers must map only immutable published artifacts: changing
/// the contents or length of the underlying file while this value is alive
/// violates `memmap2`'s file-mapping safety contract.
#[derive(Debug)]
pub struct ReadOnlyMappedFile {
    mmap: Mmap,
}

impl ReadOnlyMappedFile {
    /// Open and map an immutable published artifact read-only.
    ///
    /// The caller must uphold the repository's publication lifecycle: the file
    /// was completely written and synced before an atomic rename, and no code
    /// may mutate or truncate that published inode while readers can retain it.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be opened or the operating
    /// system cannot create a read-only mapping for it.
    pub fn open_published(path: &Path) -> io::Result<Self> {
        let path_metadata = path.symlink_metadata()?;
        if !path_metadata.file_type().is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "published artifact must be a regular file, not a symlink or special file",
            ));
        }
        let file = File::open(path)?;
        ensure_same_file(&path_metadata, &file.metadata()?)?;
        map_immutable(&file).map(|mmap| Self { mmap })
    }

    /// Borrow all mapped bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Return the mapped byte length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Return whether the mapping contains no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

#[cfg(unix)]
fn ensure_same_file(
    path_metadata: &std::fs::Metadata,
    opened: &std::fs::Metadata,
) -> io::Result<()> {
    use std::os::unix::fs::MetadataExt;

    if path_metadata.dev() != opened.dev() || path_metadata.ino() != opened.ino() {
        return Err(io::Error::other(
            "published artifact changed while it was being opened",
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_same_file(_: &std::fs::Metadata, opened: &std::fs::Metadata) -> io::Result<()> {
    if !opened.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "published artifact must be a regular file",
        ));
    }
    Ok(())
}

impl AsRef<[u8]> for ReadOnlyMappedFile {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

#[allow(unsafe_code)]
fn map_immutable(file: &File) -> io::Result<Mmap> {
    // SAFETY: This is the facade's only unsafe operation. Frankensearch maps
    // only published generation files, whose lifecycle contract forbids
    // in-place writes and truncation for as long as readers can retain them.
    // The returned Mmap owns the mapping and exposes only shared byte slices.
    unsafe { Mmap::map(file) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_exact_file_bytes_through_all_views() {
        let directory = tempfile::tempdir().expect("create temp directory");
        let path = directory.path().join("seg-0000000000000001.fslx");
        let expected = b"published immutable artifact\0with raw bytes";
        std::fs::write(&path, expected).expect("write fixture before mapping");

        let mapped = ReadOnlyMappedFile::open_published(&path).expect("map fixture");

        assert_eq!(mapped.as_bytes(), expected);
        assert_eq!(mapped.as_ref(), expected);
        assert_eq!(mapped.len(), expected.len());
        assert!(!mapped.is_empty());
    }

    #[test]
    fn reports_missing_file_as_io_error() {
        let directory = tempfile::tempdir().expect("create temp directory");
        let path = directory.path().join("missing.bin");

        let error = ReadOnlyMappedFile::open_published(&path).expect_err("missing file must fail");

        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symbolic_links() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("create temp directory");
        let target = directory.path().join("target.fslx");
        let path = directory.path().join("seg-0000000000000001.fslx");
        std::fs::write(&target, b"immutable target").expect("write target");
        symlink(&target, &path).expect("create fixture symlink");

        let error = ReadOnlyMappedFile::open_published(&path).expect_err("symlink must fail");

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn is_send_and_sync() {
        const fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<ReadOnlyMappedFile>();
    }
}
