//! The real embedder: `all-MiniLM-L6-v2` (int8-quantized ONNX) via ONNX
//! Runtime, with mean pooling over the token embeddings.
//!
//! Resolution when no explicit `--model` is requested: `$AE_MODEL_DIR` (a local
//! dir holding `model.onnx` + `tokenizer.json`) → the model fetched from the
//! HuggingFace Hub into the shared cache (`~/.cache/huggingface/hub`). If
//! nothing loads (offline + uncached), callers use the hash fallback (see
//! [`super::default_embedder`]). ONNX Runtime is statically linked by default,
//! or `dlopen`ed at runtime under the `ort-load-dynamic` feature (Homebrew) —
//! see [`ensure_ort_dylib`].

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
/// Default model on the HuggingFace Hub (ONNX int8-quantized export of
/// all-MiniLM-L6-v2). Override with `--model <dir | .onnx | org/name>`.
const DEFAULT_HF_REPO: &str = "Xenova/all-MiniLM-L6-v2";
const HF_MODEL_FILE: &str = "onnx/model_quantized.onnx";
const HF_TOKENIZER_FILE: &str = "tokenizer.json";

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
        ensure_ort_dylib();

        if let Some(spec) = spec {
            return match resolve(spec).and_then(|(m, t)| Self::from_files(&m, &t)) {
                Some(e) => Some(e),
                None => {
                    log::debug!("--model {spec}: could not load");
                    None
                }
            };
        }

        // No explicit request: an explicit local dir, else the HuggingFace Hub
        // (shared cache), else the hash fallback.
        if let Some(dir) = std::env::var_os("AE_MODEL_DIR") {
            let dir = PathBuf::from(dir);
            return Self::from_files(&dir.join("model.onnx"), &dir.join("tokenizer.json"));
        }
        let (model, tokenizer) = fetch_from_hub(DEFAULT_HF_REPO)?;
        Self::from_files(&model, &tokenizer)
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
            .map_err(|e| log::debug!("tokenizer load failed: {e}"))
            .ok()?;
        let session = build_session(model)
            .map_err(|e| log::debug!("ONNX session load failed: {e}"))
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
    fn kind(&self) -> &'static str {
        "onnx"
    }

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

/// Resolve a `--model` spec to a `(model.onnx, tokenizer.json)` pair.
///
/// A directory uses `<dir>/{model.onnx,tokenizer.json}`; a `.onnx` file pairs
/// with a sibling `tokenizer.json`; an `org/name` spec is fetched from the
/// HuggingFace Hub. Anything else doesn't resolve.
fn resolve(spec: &str) -> Option<(PathBuf, PathBuf)> {
    let path = Path::new(spec);
    if path.is_dir() {
        return Some((path.join("model.onnx"), path.join("tokenizer.json")));
    }
    if path.is_file() {
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        return Some((path.to_path_buf(), dir.join("tokenizer.json")));
    }
    // A non-path `org/name` spec is a HuggingFace repo id (don't treat a
    // mistyped/relative/absolute path as one).
    if spec.contains('/')
        && !spec.starts_with('.')
        && !spec.starts_with('/')
        && !spec.starts_with('~')
    {
        return fetch_from_hub(spec);
    }
    None
}

/// Fetch `<repo>`'s ONNX model + tokenizer from the HuggingFace Hub into the
/// shared cache (`~/.cache/huggingface/hub`, honoring `HF_HOME`), returning
/// their local paths. `None` on any error (offline + uncached → hash fallback).
fn fetch_from_hub(repo: &str) -> Option<(PathBuf, PathBuf)> {
    use hf_hub::api::sync::Api;
    let api = Api::new()
        .map_err(|e| log::debug!("HuggingFace Hub init failed: {e}"))
        .ok()?;
    let repo = api.model(repo.to_string());
    let model = repo
        .get(HF_MODEL_FILE)
        .map_err(|e| log::debug!("model fetch failed: {e}"))
        .ok()?;
    let tokenizer = repo
        .get(HF_TOKENIZER_FILE)
        .map_err(|e| log::debug!("tokenizer fetch failed: {e}"))
        .ok()?;
    Some((model, tokenizer))
}

/// Under the load-dynamic strategy, ONNX Runtime is `dlopen`ed at runtime from
/// `ORT_DYLIB_PATH`. If it isn't set, probe the usual install locations (the
/// Homebrew keg in particular) so a packaged `ae` works with no env setup. A
/// no-op for the static (download-binaries) build.
fn ensure_ort_dylib() {
    #[cfg(feature = "ort-load-dynamic")]
    {
        if std::env::var_os("ORT_DYLIB_PATH").is_some() {
            return;
        }
        let lib = if cfg!(target_os = "macos") {
            "libonnxruntime.dylib"
        } else {
            "libonnxruntime.so"
        };
        let mut bases: Vec<PathBuf> = Vec::new();
        if let Some(prefix) = std::env::var_os("HOMEBREW_PREFIX") {
            let prefix = PathBuf::from(prefix);
            bases.push(prefix.join("opt/onnxruntime/lib"));
            bases.push(prefix.join("lib"));
        }
        for p in [
            "/opt/homebrew/opt/onnxruntime/lib",
            "/opt/homebrew/lib",
            "/usr/local/lib",
            "/usr/lib",
        ] {
            bases.push(PathBuf::from(p));
        }
        if let Some(dir) = bases.into_iter().find(|d| d.join(lib).is_file()) {
            // SAFETY: called once at embedder load, before any ORT use.
            unsafe { std::env::set_var("ORT_DYLIB_PATH", dir.join(lib)) };
        }
    }
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
