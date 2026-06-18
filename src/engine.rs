//! The hybrid evaluation stack: Stage 1 expansion + Stage 2 learning.
//!
//! An [`Engine`] owns the warm state — the SQLite [`Store`], the in-memory
//! [`SharedTrie`], and an [`Embedder`] — and turns a chunk of text into an
//! [`AnalysisPayload`]. The Leader holds one of these for the process lifetime;
//! the fallback path builds an ephemeral one per invocation.

use crate::embed::{Embedder, HashEmbedder};
use crate::mrl::{compress_matryoshka_vector, cosine_similarity};
use crate::store::Store;
use crate::trie::SharedTrie;
use crate::types::{AnalysisPayload, ExpansionResult, MatchCandidate};
use crate::{learn, types::LearnedCandidate};

pub struct Engine {
    store: Store,
    trie: SharedTrie,
    embedder: Box<dyn Embedder>,
}

impl Engine {
    /// Build an engine over an existing store, hydrating the trie from it.
    pub fn new(store: Store, embedder: Box<dyn Embedder>) -> rusqlite::Result<Self> {
        let trie = SharedTrie::new();
        for acronym in store.all_acronyms()? {
            trie.insert(&acronym);
        }
        Ok(Self {
            store,
            trie,
            embedder,
        })
    }

    /// A persistent engine backed by the database at `path`, seeded with the
    /// built-in dictionary on first use.
    pub fn open(path: &std::path::Path) -> rusqlite::Result<Self> {
        let store = Store::open(path)?;
        store.seed_defaults()?;
        Self::new(store, Box::new(HashEmbedder::new()))
    }

    /// An ephemeral in-memory engine seeded with the built-in dictionary — the
    /// in-process fallback when no daemon and no persistent DB are available.
    pub fn in_memory() -> rusqlite::Result<Self> {
        let store = Store::open_in_memory()?;
        store.seed_defaults()?;
        Self::new(store, Box::new(HashEmbedder::new()))
    }

    /// Run both stages over `text` and return the unified payload.
    ///
    /// Stage 1 reads the dictionary as it stands *before* Stage 2 persists any
    /// newly learned terms, so a brand-new acronym shows up only as a learned
    /// candidate on the pass that discovers it, then as a known expansion
    /// thereafter.
    pub fn analyze(&self, text: &str) -> rusqlite::Result<AnalysisPayload> {
        let query_vec = compress_matryoshka_vector(&self.embedder.embed(text));

        let expansions = self.expand(text, &query_vec)?;
        let learned = self.learn_and_persist(text)?;

        Ok(AnalysisPayload {
            sentence: text.to_string(),
            expansions,
            learned_candidates: learned,
        })
    }

    /// Stage 1 — scan the text for known acronyms and rank their expansions.
    fn expand(&self, text: &str, query_vec: &[f32]) -> rusqlite::Result<Vec<ExpansionResult>> {
        let mut results: Vec<ExpansionResult> = Vec::new();

        for token in tokens(text) {
            if !self.trie.contains(token) {
                continue;
            }
            let acronym = token.to_uppercase();
            if results.iter().any(|r| r.acronym == acronym) {
                continue; // already handled this acronym in this sentence
            }

            let rows = self.store.expansions_for(&acronym)?;
            let n = rows.len();
            let mut matches: Vec<MatchCandidate> = rows
                .into_iter()
                .map(|(id, expansion)| {
                    let confidence = self.confidence(id, n, query_vec)?;
                    Ok(MatchCandidate {
                        expansion,
                        confidence,
                    })
                })
                .collect::<rusqlite::Result<_>>()?;

            // Best candidate first.
            matches.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

            if !matches.is_empty() {
                results.push(ExpansionResult {
                    acronym,
                    text_slice: token.to_string(),
                    matches,
                });
            }
        }
        Ok(results)
    }

    /// Confidence for one expansion: a prior (lower when the acronym is
    /// ambiguous) lifted by how well the query matches any recorded context.
    fn confidence(&self, id: i64, n_expansions: usize, query_vec: &[f32]) -> rusqlite::Result<f32> {
        let prior = if n_expansions <= 1 { 0.8 } else { 0.5 };
        let contexts = self.store.contexts_for(id)?;
        if contexts.is_empty() {
            return Ok(prior);
        }
        let best = contexts
            .iter()
            .map(|c| cosine_similarity(query_vec, c))
            .fold(0.0_f32, f32::max)
            .clamp(0.0, 1.0);
        // Evidence can only raise confidence above the floor.
        Ok((0.5 + 0.5 * best).max(prior).min(1.0))
    }

    /// Stage 2 — extract inline definitions, persist them (dictionary + trie +
    /// a context embedding), and return them.
    fn learn_and_persist(&self, text: &str) -> rusqlite::Result<Vec<LearnedCandidate>> {
        let learned = learn::extract(text);
        for c in &learned {
            let id = self.store.add_entry(&c.acronym, &c.extracted_definition)?;
            self.trie.insert(&c.acronym);
            let ctx = compress_matryoshka_vector(&self.embedder.embed(&c.extracted_definition));
            self.store.add_context(id, &ctx)?;
        }
        Ok(learned)
    }

    /// Number of acronyms currently known — exposed for diagnostics/tests.
    pub fn known_acronyms(&self) -> usize {
        self.trie.len()
    }
}

/// Split text into alphanumeric tokens, preserving each token's original
/// spelling (so `text_slice` reflects what the user wrote).
fn tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_a_seeded_acronym() {
        let e = Engine::in_memory().unwrap();
        let out = e.analyze("Check the OKR dashboard.").unwrap();
        assert_eq!(out.expansions.len(), 1);
        assert_eq!(out.expansions[0].acronym, "OKR");
        assert_eq!(
            out.expansions[0].matches[0].expansion,
            "Objectives and Key Results"
        );
    }

    #[test]
    fn learns_a_novel_acronym_then_can_expand_it() {
        let e = Engine::in_memory().unwrap();

        // First pass: ZQ is unknown, so it's only a learned candidate.
        let first = e.analyze("The ZQ (Zebra Queue) is deep.").unwrap();
        assert!(first.learned_candidates.iter().any(|c| c.acronym == "ZQ"));
        assert!(!first.expansions.iter().any(|r| r.acronym == "ZQ"));

        // Second pass: now it's known and expands.
        let second = e.analyze("Drain the ZQ now.").unwrap();
        let zq = second
            .expansions
            .iter()
            .find(|r| r.acronym == "ZQ")
            .unwrap();
        assert_eq!(zq.matches[0].expansion, "Zebra Queue");
    }

    #[test]
    fn text_slice_preserves_original_casing() {
        let e = Engine::in_memory().unwrap();
        let out = e.analyze("look at the okr").unwrap();
        let r = out.expansions.iter().find(|r| r.acronym == "OKR").unwrap();
        assert_eq!(r.text_slice, "okr");
    }

    #[test]
    fn confidence_stays_in_unit_range() {
        let e = Engine::in_memory().unwrap();
        let out = e.analyze("KPI and OKR and API").unwrap();
        for r in &out.expansions {
            for m in &r.matches {
                assert!(
                    m.confidence > 0.0 && m.confidence <= 1.0,
                    "{}",
                    m.confidence
                );
            }
        }
    }

    #[test]
    fn plain_text_produces_an_empty_payload() {
        let e = Engine::in_memory().unwrap();
        let out = e.analyze("nothing notable here").unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn an_acronym_is_reported_once_per_sentence() {
        let e = Engine::in_memory().unwrap();
        let out = e.analyze("OKR here, OKR there, OKR everywhere").unwrap();
        assert_eq!(
            out.expansions.iter().filter(|r| r.acronym == "OKR").count(),
            1
        );
    }
}
