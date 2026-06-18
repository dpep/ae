//! Build-time acquisition of the embedding model.
//!
//! The quantized ONNX model is *not* checked into the repo — it's fetched once
//! at build time into a persistent user cache (so rebuilds reuse it, no
//! re-download). We download the upstream **int8-quantized** export of
//! all-MiniLM-L6-v2 (~22 MB) — that's the "shrink": running a quantizer locally
//! would need Python's onnxruntime tooling, which we deliberately avoid.
//!
//! Two consumption modes:
//! - **bundled** (default `bundled-model` feature): the model is staged into
//!   `OUT_DIR` and `include_bytes!`'d into the binary → one self-contained file.
//! - **external**: the cache path is baked in as `AE_MODEL_DIR` and loaded at
//!   runtime from disk.
//!
//! Everything is best-effort: if the download fails (offline, sandbox,
//! `AE_SKIP_MODEL_DOWNLOAD=1`), the build still succeeds — a bundled build gets
//! empty placeholders and the binary falls back to the hash embedder.
//!
//! Overridable via env: `AE_CACHE_DIR`, `AE_MODEL_URL`, `AE_TOKENIZER_URL`,
//! `AE_SKIP_MODEL_DOWNLOAD`.

use std::path::{Path, PathBuf};
use std::process::Command;

const MODEL_VERSION: &str = "all-MiniLM-L6-v2-quantized";
const DEFAULT_MODEL_URL: &str =
    "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx";
const DEFAULT_TOKENIZER_URL: &str =
    "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for var in [
        "AE_CACHE_DIR",
        "AE_MODEL_URL",
        "AE_TOKENIZER_URL",
        "AE_SKIP_MODEL_DOWNLOAD",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let cache = cache_dir();
    let _ = std::fs::create_dir_all(&cache);
    let model = cache.join("model.onnx");
    let tokenizer = cache.join("tokenizer.json");

    let have = if std::env::var_os("AE_SKIP_MODEL_DOWNLOAD").is_some() {
        model.is_file() && tokenizer.is_file()
    } else {
        ensure(&model, &url("AE_MODEL_URL", DEFAULT_MODEL_URL))
            && ensure(&tokenizer, &url("AE_TOKENIZER_URL", DEFAULT_TOKENIZER_URL))
    };

    if have {
        // External-loading mode resolves the model from here.
        println!("cargo:rustc-env=AE_MODEL_DIR={}", cache.display());
    } else {
        warn("embedding model unavailable; binary will use the hash embedder");
    }

    // Bundled mode: stage the asset into OUT_DIR for `include_bytes!`. The files
    // must always exist (empty placeholder if absent) so the include compiles.
    if std::env::var_os("CARGO_FEATURE_BUNDLED_MODEL").is_some() {
        let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));
        stage(&model, &out.join("model.onnx"), have);
        stage(&tokenizer, &out.join("tokenizer.json"), have);
    }
}

/// Persistent cache dir, keyed by model version so a version bump re-downloads.
fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("AE_CACHE_DIR") {
        return PathBuf::from(dir).join(MODEL_VERSION);
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("ae").join("models").join(MODEL_VERSION)
}

fn url(env_key: &str, default: &str) -> String {
    std::env::var(env_key).unwrap_or_else(|_| default.to_string())
}

/// Download `url` to `path` unless a non-empty file is already cached.
fn ensure(path: &Path, url: &str) -> bool {
    if path.is_file() && path.metadata().map(|m| m.len() > 0).unwrap_or(false) {
        return true;
    }
    let tmp = path.with_extension("part");
    let _ = std::fs::remove_file(&tmp);
    let status = Command::new("curl")
        .args(["-sSL", "--fail", "--retry", "2", "-o"])
        .arg(&tmp)
        .arg(url)
        .status();
    match status {
        Ok(s) if s.success() => std::fs::rename(&tmp, path).is_ok(),
        _ => {
            let _ = std::fs::remove_file(&tmp);
            false
        }
    }
}

/// Copy `src` → `dst` for bundling, or write an empty placeholder when the model
/// isn't available (keeps the `include_bytes!` compiling offline).
fn stage(src: &Path, dst: &Path, have: bool) {
    let copied = have && std::fs::copy(src, dst).is_ok();
    if !copied {
        let _ = std::fs::write(dst, []);
    }
}

fn warn(msg: &str) {
    println!("cargo:warning=ae: {msg}");
}
