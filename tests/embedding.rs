//! Real-model embedding checks. These exercise the ONNX path end-to-end, so
//! they only run when the model is actually present (it's fetched at build
//! time); offline builds that fell back to the hash embedder skip them rather
//! than fail.

use ae::embed::{Embedder, OnnxEmbedder};
use ae::mrl::{compress_matryoshka_vector, cosine_similarity};

/// Compress through the same MRL pipeline the engine uses.
fn vec_of(e: &OnnxEmbedder, text: &str) -> Vec<f32> {
    compress_matryoshka_vector(&e.embed(text))
}

#[test]
fn semantically_related_text_is_closer_than_unrelated() {
    let Some(e) = OnnxEmbedder::load(None) else {
        eprintln!("skipping: embedding model unavailable");
        return;
    };

    let anchor = vec_of(&e, "a small domestic cat");
    let related = vec_of(&e, "a tiny kitten");
    let unrelated = vec_of(&e, "quarterly revenue and profit margins");

    let near = cosine_similarity(&anchor, &related);
    let far = cosine_similarity(&anchor, &unrelated);
    assert!(
        near > far,
        "related {near:.3} should exceed unrelated {far:.3}"
    );
}

#[test]
fn the_compressed_vector_is_64_dims_and_unit_norm() {
    let Some(e) = OnnxEmbedder::load(None) else {
        eprintln!("skipping: embedding model unavailable");
        return;
    };
    let v = vec_of(&e, "Key Performance Indicator");
    assert_eq!(v.len(), ae::mrl::MRL_DIMS);
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "norm was {norm}");
}

#[test]
fn embedding_is_deterministic() {
    let Some(e) = OnnxEmbedder::load(None) else {
        eprintln!("skipping: embedding model unavailable");
        return;
    };
    assert_eq!(
        e.embed("Application Programming Interface"),
        e.embed("Application Programming Interface")
    );
}
