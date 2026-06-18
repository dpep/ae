//! The real embedder: `all-MiniLM-L6-v2` (int8-quantized ONNX) via ONNX
//! Runtime, with mean pooling over the token embeddings.
//!
//! The model and tokenizer are loaded from bytes — either baked into the binary
//! (default `bundled-model` feature → one self-contained file) or read from
//! disk (dev/test, or an explicit `--model`). Resolution order when no explicit
//! model is requested: `$AE_MODEL_DIR` (runtime override) → bundled bytes →
//! the build-time cache path. If nothing loads, callers use the hash fallback
//! (see [`super::default_embedder`]).

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use super::Embedder;

/// all-MiniLM-L6-v2 hidden size.
const NATIVE_DIMS: usize = 384;
/// Cap sequence length — short jargon phrases never need the full context.
const MAX_SEQ: usize = 256;

#[cfg(feature = "bundled-model")]
const BUNDLED_MODEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/model.onnx"));
#[cfg(feature = "bundled-model")]
const BUNDLED_TOKENIZER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tokenizer.json"));

pub struct OnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
}

impl OnnxEmbedder {
    /// Load the embedder. `spec` is an optional `--model` request: an absolute
    /// or relative path (to a model directory or `.onnx` file) or a bare name
    /// resolved against the model search dirs. Returns `None` (→ hash fallback)
    /// if nothing usable loads.
    pub fn load(spec: Option<&str>) -> Option<Self> {
        if let Some(spec) = spec {
            return match resolve(spec).and_then(|(m, t)| Self::from_files(&m, &t)) {
                Some(e) => Some(e),
                None => {
                    log::warn!("--model {spec}: could not load; using hash embedder");
                    None
                }
            };
        }

        // No explicit request: runtime override → bundled → build-time cache.
        if let Some(dir) = std::env::var_os("AE_MODEL_DIR") {
            let dir = PathBuf::from(dir);
            return Self::from_files(&dir.join("model.onnx"), &dir.join("tokenizer.json"));
        }
        if let Some(e) = bundled() {
            return Some(e);
        }
        let baked = PathBuf::from(option_env!("AE_MODEL_DIR")?);
        Self::from_files(&baked.join("model.onnx"), &baked.join("tokenizer.json"))
    }

    fn from_files(model: &Path, tokenizer: &Path) -> Option<Self> {
        let model = std::fs::read(model).ok()?;
        let tokenizer = std::fs::read(tokenizer).ok()?;
        Self::from_bytes(&model, &tokenizer)
    }

    fn from_bytes(model: &[u8], tokenizer: &[u8]) -> Option<Self> {
        if model.is_empty() {
            return None; // empty placeholder from an offline bundled build
        }
        let tokenizer = Tokenizer::from_bytes(tokenizer)
            .map_err(|e| log::warn!("tokenizer load failed: {e}"))
            .ok()?;
        let session = build_session(model)
            .map_err(|e| log::warn!("ONNX session load failed: {e}"))
            .ok()?;
        Some(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }

    pub fn dims(&self) -> usize {
        NATIVE_DIMS
    }

    /// Tokenize, run the model, and mean-pool — `None` on any error so the
    /// public [`Embedder::embed`] can degrade to a zero vector.
    fn run(&self, text: &str) -> Option<Vec<f32>> {
        // all-MiniLM-L6-v2 uses no task prefix — encode the text directly.
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| log::warn!("tokenize failed: {e}"))
            .ok()?;

        let take = encoding.get_ids().len().min(MAX_SEQ);
        let ids: Vec<i64> = encoding.get_ids()[..take]
            .iter()
            .map(|&x| x as i64)
            .collect();
        let mask: Vec<i64> = encoding.get_attention_mask()[..take]
            .iter()
            .map(|&x| x as i64)
            .collect();
        let type_ids = vec![0i64; take];
        let shape = [1_i64, take as i64];

        let ids_t = Tensor::from_array((shape, ids)).ok()?;
        let mask_t = Tensor::from_array((shape, mask.clone())).ok()?;
        let type_t = Tensor::from_array((shape, type_ids)).ok()?;

        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids_t,
                "attention_mask" => mask_t,
                "token_type_ids" => type_t,
            ])
            .map_err(|e| log::warn!("inference failed: {e}"))
            .ok()?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>().ok()?;
        let seq = shape[1] as usize;
        let hidden = shape[2] as usize;
        Some(mean_pool(data, &mask, seq, hidden))
    }
}

impl Embedder for OnnxEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        self.run(text).unwrap_or_else(|| vec![0.0; NATIVE_DIMS])
    }
}

/// Build the inference session from model bytes with consistent settings.
fn build_session(model: &[u8]) -> ort::Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .commit_from_memory(model)
}

/// The model baked into the binary, if compiled with the `bundled-model`
/// feature and the asset was actually present at build time.
fn bundled() -> Option<OnnxEmbedder> {
    #[cfg(feature = "bundled-model")]
    {
        OnnxEmbedder::from_bytes(BUNDLED_MODEL, BUNDLED_TOKENIZER)
    }
    #[cfg(not(feature = "bundled-model"))]
    {
        None
    }
}

/// Resolve a `--model` spec to a `(model.onnx, tokenizer.json)` pair.
///
/// A path to a directory uses `<dir>/{model.onnx,tokenizer.json}`; a path to a
/// `.onnx` file pairs it with a sibling `tokenizer.json`; anything else is a
/// bare name looked up under the model search dirs.
fn resolve(spec: &str) -> Option<(PathBuf, PathBuf)> {
    let path = Path::new(spec);
    if path.is_dir() {
        return Some((path.join("model.onnx"), path.join("tokenizer.json")));
    }
    if path.is_file() {
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        return Some((path.to_path_buf(), dir.join("tokenizer.json")));
    }
    for base in search_dirs() {
        let dir = base.join(spec);
        if dir.join("model.onnx").is_file() && dir.join("tokenizer.json").is_file() {
            return Some((dir.join("model.onnx"), dir.join("tokenizer.json")));
        }
    }
    None
}

/// Directories searched for a named model: `$AE_MODELS_DIR`, the user cache,
/// and a dir alongside the executable (Homebrew installs models there).
fn search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::var_os("AE_MODELS_DIR") {
        dirs.push(PathBuf::from(d));
    }
    let cache_base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir);
    dirs.push(cache_base.join("ae").join("models"));
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        dirs.push(dir.join("../share/ae/models"));
    }
    dirs
}

/// Mean-pool `[1, seq, hidden]` token embeddings, weighted by the attention
/// mask, into one `hidden`-length vector.
fn mean_pool(data: &[f32], mask: &[i64], seq: usize, hidden: usize) -> Vec<f32> {
    let mut pooled = vec![0.0f32; hidden];
    let mut count = 0.0f32;
    for t in 0..seq {
        if mask.get(t).copied().unwrap_or(0) == 0 {
            continue;
        }
        count += 1.0;
        let row = &data[t * hidden..(t + 1) * hidden];
        for (p, &x) in pooled.iter_mut().zip(row) {
            *p += x;
        }
    }
    if count > 0.0 {
        for p in &mut pooled {
            *p /= count;
        }
    }
    pooled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_pool_ignores_masked_positions() {
        // Two tokens, hidden=2; second token masked out → equals the first row.
        let data = [1.0, 2.0, 9.0, 9.0];
        let mask = [1, 0];
        assert_eq!(mean_pool(&data, &mask, 2, 2), vec![1.0, 2.0]);
    }

    #[test]
    fn mean_pool_averages_active_positions() {
        let data = [1.0, 1.0, 3.0, 3.0];
        let mask = [1, 1];
        assert_eq!(mean_pool(&data, &mask, 2, 2), vec![2.0, 2.0]);
    }

    #[test]
    fn a_bare_name_with_no_matching_dir_does_not_resolve() {
        assert!(resolve("definitely-not-a-real-model-name-xyz").is_none());
    }

    #[test]
    fn a_nonexistent_path_spec_yields_no_embedder() {
        let missing = std::env::temp_dir().join(format!("ae-missing-{}", std::process::id()));
        assert!(OnnxEmbedder::load(Some(missing.to_str().unwrap())).is_none());
    }
}
