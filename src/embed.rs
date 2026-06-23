//! Text embedding behind a trait, with two implementations.
//!
//! [`OnnxEmbedder`] runs the real `all-MiniLM-L6-v2` model (int8-quantized
//! ONNX, fetched from the HuggingFace Hub on first use — see [`onnx`]) via ONNX
//! Runtime. It's the default when the model loads. [`HashEmbedder`] is a
//! deterministic, dependency-free feature-hash fallback used when the model
//! can't be loaded (offline + uncached, missing asset) and in unit tests that
//! need reproducibility.
//!
//! Both yield a native-width vector that the MRL pipeline truncates to 64 dims;
//! the trait deliberately does *not* fix the width, since the implementations
//! happen to agree here ([`EMBED_DIMS`] = 384) but needn't. [`default_embedder`]
//! picks the best available at runtime, so callers never branch on which one
//! they got.

mod onnx;

pub use onnx::OnnxEmbedder;

/// Native width of the [`HashEmbedder`] fallback. The MRL stage only needs at
/// least [`crate::mrl::MRL_DIMS`] coordinates, so embedders may differ in width.
pub const EMBED_DIMS: usize = 384;

/// Produces an embedding (length ≥ [`crate::mrl::MRL_DIMS`]) for a chunk of text.
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Vec<f32>;
}

/// The best embedder available: the ONNX model if it loads, else the hash
/// fallback. `model` is an optional `--model` request (path or name). Logs
/// which embedder was selected.
pub fn default_embedder(model: Option<&str>) -> Box<dyn Embedder> {
    match OnnxEmbedder::load(model) {
        Some(e) => {
            log::info!("using ONNX embedder ({} dims)", e.dims());
            Box::new(e)
        }
        None => {
            // The ONNX model (or its tokenizer) didn't load — surface it once,
            // clearly, since semantic matching is degraded until it's fixed. The
            // underlying cause is logged at debug (run with -v / RUST_LOG=debug).
            log::warn!(
                "embedding model unavailable — using the built-in hash fallback \
                 (reduced semantic accuracy). Pass --model <dir|.onnx|org/name>, \
                 or run with -v to see why the model didn't load."
            );
            Box::new(HashEmbedder::new())
        }
    }
}

/// Deterministic, dependency-free embedder via signed feature hashing over
/// word tokens. Not semantically trained, but stable and similarity-preserving
/// at the lexical level — enough to exercise and test the vector pipeline.
#[derive(Default, Clone, Copy, Debug)]
pub struct HashEmbedder;

impl HashEmbedder {
    pub fn new() -> Self {
        Self
    }
}

/// FNV-1a — a small, fast, deterministic hash (no RNG, stable across runs).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Split into lowercased alphanumeric tokens.
fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; EMBED_DIMS];
        for token in tokenize(text) {
            let h = fnv1a(token.as_bytes());
            let idx = (h % EMBED_DIMS as u64) as usize;
            // A second hashed bit picks the sign, reducing collisions' bias.
            let sign = if (h >> 32) & 1 == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_has_full_width() {
        assert_eq!(HashEmbedder::new().embed("hello world").len(), EMBED_DIMS);
    }

    #[test]
    fn embedding_is_deterministic() {
        let e = HashEmbedder::new();
        assert_eq!(
            e.embed("Key Performance Indicator"),
            e.embed("Key Performance Indicator")
        );
    }

    #[test]
    fn is_case_and_punctuation_insensitive() {
        let e = HashEmbedder::new();
        assert_eq!(
            e.embed("Key Performance Indicator"),
            e.embed("key, performance. indicator!")
        );
    }

    #[test]
    fn different_text_gives_different_vectors() {
        let e = HashEmbedder::new();
        assert_ne!(e.embed("apples"), e.embed("oranges"));
    }

    #[test]
    fn empty_text_is_the_zero_vector() {
        assert_eq!(HashEmbedder::new().embed(""), vec![0.0; EMBED_DIMS]);
    }
}
