//! Matryoshka Representation Learning (MRL) vector compression.
//!
//! MRL-trained embeddings keep their most significant information in the
//! leading coordinates, so a high-dimensional vector can be truncated to a
//! short prefix and still retrieve well. We truncate to 64 dimensions and
//! L2-normalize, which shrinks the stored vector ~6× versus a 384-d float
//! payload while keeping the bulk of the semantic signal.

/// The compressed embedding width every stored/queried vector is reduced to.
pub const MRL_DIMS: usize = 64;

/// Truncate `raw_embedding` to the leading [`MRL_DIMS`] coordinates and
/// L2-normalize it onto the unit sphere.
///
/// A zero (or shorter-than-`MRL_DIMS`-after-truncation) vector can't be
/// normalized, so the truncated values are returned unchanged in that case to
/// avoid dividing by zero.
///
/// # Panics
/// Panics if `raw_embedding` has fewer than [`MRL_DIMS`] elements — callers
/// must pass a full-width embedding.
pub fn compress_matryoshka_vector(raw_embedding: &[f32]) -> Vec<f32> {
    assert!(
        raw_embedding.len() >= MRL_DIMS,
        "embedding has {} dims, need at least {MRL_DIMS}",
        raw_embedding.len()
    );

    let truncated = &raw_embedding[0..MRL_DIMS];

    let norm = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        truncated.iter().map(|x| x / norm).collect()
    } else {
        truncated.to_vec()
    }
}

/// Cosine similarity between two equal-length vectors.
///
/// For unit-norm inputs (what [`compress_matryoshka_vector`] produces) this is
/// just the dot product, but we divide by the norms anyway so the function is
/// correct for arbitrary inputs. Returns `0.0` if either vector is degenerate
/// or the lengths differ.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(len: usize) -> Vec<f32> {
        (0..len).map(|i| i as f32 + 1.0).collect()
    }

    #[test]
    fn truncates_to_64_dims() {
        let out = compress_matryoshka_vector(&ramp(384));
        assert_eq!(out.len(), MRL_DIMS);
    }

    #[test]
    fn output_is_unit_norm() {
        let out = compress_matryoshka_vector(&ramp(384));
        let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    #[test]
    fn zero_vector_is_returned_unnormalized() {
        let out = compress_matryoshka_vector(&vec![0.0; 64]);
        assert_eq!(out, vec![0.0; 64]);
    }

    #[test]
    #[should_panic]
    fn rejects_too_short_input() {
        compress_matryoshka_vector(&ramp(10));
    }

    #[test]
    fn cosine_of_identical_unit_vectors_is_one() {
        let v = compress_matryoshka_vector(&ramp(64));
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_zero() {
        let mut a = vec![0.0; 64];
        let mut b = vec![0.0; 64];
        a[0] = 1.0;
        b[1] = 1.0;
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_mismatched_lengths_is_zero() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
    }
}
