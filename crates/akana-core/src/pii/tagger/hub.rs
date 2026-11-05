//! The tagger's models on the Hugging Face Hub ([`REPO`]): downloaded on first use into a local
//! cache, from a pinned commit of the repository ([`REVISION`]), and checked against the sizes and
//! SHA-256 sums pinned here before they are used.
//!
//! The cache is `$AKANA_CACHE_DIR` when it is set, otherwise `akana` in the platform's cache
//! directory (`%LOCALAPPDATA%` on Windows, `~/Library/Caches` on macOS, `$XDG_CACHE_HOME` or
//! `~/.cache` elsewhere); the files go under `pii-tagger/<revision>/`. `HF_ENDPOINT` points the
//! downloads at a mirror, and `AKANA_OFFLINE=1` or `HF_HUB_OFFLINE=1` turns them off, so that a
//! model missing from the cache is an error.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use sha2::{Digest, Sha256};

use super::{Mode, PiiTagger, TaggerError, TaggerOptions};

/// The model repository on the Hugging Face Hub.
pub const REPO: &str = "altaidevorg/akana-pii-tagger";
/// The repository commit the files come from.
pub const REVISION: &str = "03dfe94bc1d672325caf2f9254b645089ffbd5ef";

/// A file of the repository, with its size in bytes and its SHA-256.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Asset {
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// The small model: the fast mode, and the accurate mode's first member.
pub const SMALL: [Asset; 2] = [
    Asset {
        path: "electra-small/model.onnx",
        size: 56_557_544,
        sha256: "5fe09d67fdb1c3b7f6d3b16929e001e5b485e0fdf0db8a7b219b47c2d153bec0",
    },
    Asset {
        path: "electra-small/tokenizer.json",
        size: 754_625,
        sha256: "27548621792d11fd7890ed47815802065c1d8eacf18bbc3513ee5e4b65a6fc45",
    },
];

/// BERTurk, averaged with the small model on every window in the accurate mode.
pub const BIG: [Asset; 2] = [
    Asset {
        path: "berturk/model.onnx",
        size: 441_897_268,
        sha256: "93a8df10104d086abe3aa8924471024383ec2c210fbdd056fe9a1d165741d931",
    },
    Asset {
        path: "berturk/tokenizer.json",
        size: 754_624,
        sha256: "4251d23347259b238ea1697f55e182f55106584e76fdec86efd2e4b441157ab0",
    },
];

/// A download that has not moved for this long is given up.
const STALL: Duration = Duration::from_secs(60);
/// A whole file must arrive within this time (441 MB at about 125 kB/s).
const BODY_BUDGET: Duration = Duration::from_secs(3600);
/// Partial files older than this are left over from an interrupted download, and are removed.
const STALE_PART: Duration = Duration::from_secs(24 * 3600);

impl PiiTagger {
    /// Loads the tagger of `mode` from the cache, downloading its models first if they are not
    /// there yet (see the module's documentation).
    pub fn from_hub(mode: Mode, options: TaggerOptions) -> Result<Self, TaggerError> {
        let dirs = model_dirs(mode, &mut |_, _| {})?;
        Self::load(&dirs, options)
    }
}

/// The directory that holds this revision's models (it may not exist yet).
pub fn cache_dir() -> Result<PathBuf, TaggerError> {
    let root = match env_path("AKANA_CACHE_DIR") {
        Some(dir) => dir,
        None => platform_cache_dir()
            .ok_or(TaggerError::NoCacheDir)?
            .join("akana"),
    };
    Ok(root.join("pii-tagger").join(REVISION))
}

/// The model directories of `mode` in the cache, the small model's first, downloading the files
/// the cache lacks. `on_download` is told each file's repository path and size before it is fetched.
pub fn model_dirs(
    mode: Mode,
    on_download: &mut dyn FnMut(&str, u64),
) -> Result<Vec<PathBuf>, TaggerError> {
    let root = cache_dir()?;
    let members: &[&[Asset]] = match mode {
        Mode::Fast => &[&SMALL],
        Mode::Accurate => &[&SMALL, &BIG],
    };
    let mut agent = None;
    let mut dirs = Vec::with_capacity(members.len());
    for assets in members {
        for asset in assets.iter() {
            let dest = root.join(asset.path);
            if is_cached(&dest, asset) {
                continue;
            }
            if offline() {
                return Err(TaggerError::Offline(dest));
            }
            on_download(asset.path, asset.size);
            fetch(agent.get_or_insert_with(new_agent), asset, &dest)?;
        }
        let dir = assets
            .first()
            .and_then(|a| Path::new(a.path).parent())
            .ok_or(TaggerError::NoModel)?;
        dirs.push(root.join(dir));
    }
    Ok(dirs)
}

/// A file counts as cached when it has the pinned size: files are moved into place only after
/// their SHA-256 has been checked.
fn is_cached(dest: &Path, asset: &Asset) -> bool {
    fs::metadata(dest).is_ok_and(|m| m.is_file() && m.len() == asset.size)
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn platform_cache_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        env_path("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        env_path("HOME").map(|home| home.join("Library").join("Caches"))
    } else {
        env_path("XDG_CACHE_HOME")
            .filter(|dir| dir.is_absolute())
            .or_else(|| env_path("HOME").map(|home| home.join(".cache")))
    }
}

fn offline() -> bool {
    ["AKANA_OFFLINE", "HF_HUB_OFFLINE"]
        .iter()
        .any(|name| std::env::var(name).is_ok_and(|v| is_true(&v)))
}

/// The values Hugging Face's libraries read as true in their environment flags.
fn is_true(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .map(|e| e.trim().trim_end_matches('/').to_string())
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string())
}

fn new_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(STALL))
        .timeout_recv_response(Some(STALL))
        .timeout_recv_body(Some(BODY_BUDGET))
        .user_agent(concat!("akana/", env!("CARGO_PKG_VERSION")))
        .build()
        .into()
}

/// Downloads `asset` to `dest`: into a partial file beside it, checked, then moved into place, so
/// that `dest` never holds a partial or unchecked file (two processes fetching at once each write
/// their own partial file).
fn fetch(agent: &ureq::Agent, asset: &Asset, dest: &Path) -> Result<(), TaggerError> {
    let url = format!("{}/{REPO}/resolve/{REVISION}/{}", endpoint(), asset.path);
    let (Some(dir), Some(name)) = (dest.parent(), dest.file_name()) else {
        return Err(TaggerError::NoCacheDir);
    };
    fs::create_dir_all(dir).map_err(|source| TaggerError::Io {
        path: dir.to_path_buf(),
        source,
    })?;
    remove_stale_parts(dir);
    let part = dir.join(format!(
        "{}.{}.part",
        name.to_string_lossy(),
        std::process::id()
    ));
    let fetched = download(agent, &url, &part, asset);
    if fetched.is_err() {
        let _ = fs::remove_file(&part);
    }
    fetched?;
    fs::rename(&part, dest).map_err(|source| {
        let _ = fs::remove_file(&part);
        TaggerError::Io {
            path: dest.to_path_buf(),
            source,
        }
    })
}

fn download(agent: &ureq::Agent, url: &str, part: &Path, asset: &Asset) -> Result<(), TaggerError> {
    let failed = |message: String| TaggerError::Download {
        url: url.to_string(),
        message,
    };
    let mut response = agent.get(url).call().map_err(|e| failed(e.to_string()))?;
    let file = fs::File::create(part).map_err(|source| TaggerError::Io {
        path: part.to_path_buf(),
        source,
    })?;
    let mut sink = Hashing::new(io::BufWriter::new(file), asset.size);
    io::copy(&mut response.body_mut().as_reader(), &mut sink).map_err(|e| failed(e.to_string()))?;
    let (writer, len, digest) = sink.finish();
    let file = writer.into_inner().map_err(|e| TaggerError::Io {
        path: part.to_path_buf(),
        source: e.into_error(),
    })?;
    file.sync_all().map_err(|source| TaggerError::Io {
        path: part.to_path_buf(),
        source,
    })?;
    if len != asset.size || digest != asset.sha256 {
        return Err(TaggerError::Integrity {
            path: part.to_path_buf(),
            message: format!(
                "{url} gave {len} bytes with SHA-256 {digest}; expected {} bytes with SHA-256 {}",
                asset.size, asset.sha256
            ),
        });
    }
    Ok(())
}

/// Removes partial files that an interrupted download left behind.
fn remove_stale_parts(dir: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let path = entry.path();
        let is_part = path.extension().is_some_and(|e| e == "part");
        let age = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| now.duration_since(t).ok());
        if is_part && age.is_some_and(|a| a > STALE_PART) {
            let _ = fs::remove_file(path);
        }
    }
}

/// A writer that hashes what it writes, counts it, and refuses more than `limit` bytes.
struct Hashing<W> {
    inner: W,
    hasher: Sha256,
    len: u64,
    limit: u64,
}

impl<W: Write> Hashing<W> {
    fn new(inner: W, limit: u64) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
            len: 0,
            limit,
        }
    }

    /// The writer, the byte count and the lower-case hex SHA-256.
    fn finish(self) -> (W, u64, String) {
        let digest = self
            .hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        (self.inner, self.len, digest)
    }
}

impl<W: Write> Write for Hashing<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.len + buf.len() as u64 > self.limit {
            return Err(io::Error::other(format!(
                "more than the expected {} bytes",
                self.limit
            )));
        }
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.len += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_hex(s: &str, len: usize) -> bool {
        s.len() == len
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    #[test]
    fn the_pinned_files_are_well_formed() {
        assert!(is_hex(REVISION, 40), "REVISION is not a commit hash");
        for (assets, dir) in [(&SMALL, "electra-small/"), (&BIG, "berturk/")] {
            for asset in assets {
                assert!(asset.path.starts_with(dir), "{}", asset.path);
                assert!(asset.size > 0, "{} has no size", asset.path);
                assert!(is_hex(asset.sha256, 64), "{} has no SHA-256", asset.path);
            }
            let names: Vec<_> = assets
                .iter()
                .filter_map(|a| Path::new(a.path).file_name())
                .collect();
            assert!(
                names.iter().any(|n| *n == "model.onnx")
                    && names.iter().any(|n| *n == "tokenizer.json")
            );
        }
    }

    #[test]
    fn hashes_and_counts_what_it_writes() {
        let mut sink = Hashing::new(Vec::new(), 3);
        sink.write_all(b"abc").unwrap();
        assert!(
            sink.write_all(b"d").is_err(),
            "a fourth byte must be refused"
        );
        let (written, len, digest) = sink.finish();
        assert_eq!((written.as_slice(), len), (&b"abc"[..], 3));
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn a_file_counts_as_cached_only_at_its_pinned_size() {
        let dir = std::env::temp_dir().join(format!("akana-hub-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.onnx");
        fs::write(&path, b"12345").unwrap();
        let asset = |size| Asset {
            path: "electra-small/model.onnx",
            size,
            sha256: "",
        };
        assert!(is_cached(&path, &asset(5)));
        assert!(!is_cached(&path, &asset(6)));
        assert!(!is_cached(&dir.join("missing.onnx"), &asset(5)));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reads_flags_as_hugging_face_does() {
        for v in ["1", "true", "TRUE", "yes", "On", " 1 "] {
            assert!(is_true(v), "{v}");
        }
        for v in ["0", "", "false", "no", "off", "2"] {
            assert!(!is_true(v), "{v}");
        }
    }
}
