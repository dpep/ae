//! Text embedding behind a trait so the heavy model is optional.
//!
//! The spec calls for a local ONNX model (`nomic-embed-text-v1.5`) producing
//! 384-d embeddings. That model is a large binary asset and the runtime is a
//! heavy native dependency, neither of which fits the "lightweight, runs in a
//! clean checkout" goal — so it's deferred behind a future `onnx` feature.
//!
//! The default [`HashEmbedder`] is a deterministic feature-hash that produces a
//! real [`EMBED_DIMS`]-d vector from text: similar token sets yield similar
//! vectors, with no model file and no network. That keeps the whole MRL
//! pipeline (truncate → normalize → cosine) genuine and testable. A real model
//! implements the same [`Embedder`] trait, so callers never change.

/// Native embedding width before MRL compression (matches the target model).
pub const EMBED_DIMS: usize = 384;

/// Produces a fixed-width embedding for a chunk of text.
pub trait Embedder: Send + Sync {
    /// Returns an [`EMBED_DIMS`]-length vector.
    fn embed(&self, text: &str) -> Vec<f32>;
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
