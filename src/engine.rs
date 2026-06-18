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
use crate::{learn, types::Extraction};

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
    /// built-in dictionary on first use. `model` is an optional `--model`
    /// request; otherwise the best available embedder is chosen (see
    /// [`crate::embed::default_embedder`]).
    pub fn open(path: &std::path::Path, model: Option<&str>) -> rusqlite::Result<Self> {
        let store = Store::open(path)?;
        store.seed_defaults()?;
        Self::new(store, crate::embed::default_embedder(model))
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
        let candidates = candidate_acronyms(text, &expansions, &learned);

        // Track how often we see undefined acronyms, and mine the text for
        // speculative expansions (phrases whose initials spell them). Analysis
        // only — read-only leaves no trace.
        for candidate in &candidates {
            self.store.record_candidate(candidate)?;
            for phrase in mine_potentials(text, candidate) {
                self.store.record_potential(candidate, &phrase)?;
            }
        }

        Ok(AnalysisPayload {
            sentence: text.to_string(),
            expansions,
            extractions: learned,
            candidates,
        })
    }

    /// Read-only Stage 1 only: expand known acronyms (and flag unknown ones)
    /// without extracting or persisting anything. The dictionary is never
    /// written, so this is safe to run against a shared DB without side effects.
    pub fn expand_only(&self, text: &str) -> rusqlite::Result<AnalysisPayload> {
        let query_vec = compress_matryoshka_vector(&self.embedder.embed(text));
        let expansions = self.expand(text, &query_vec)?;
        let candidates = candidate_acronyms(text, &expansions, &[]);
        Ok(AnalysisPayload {
            sentence: text.to_string(),
            expansions,
            extractions: Vec::new(),
            candidates,
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
    fn learn_and_persist(&self, text: &str) -> rusqlite::Result<Vec<Extraction>> {
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

    /// Candidate acronyms and their occurrence counts (seen but undefined).
    pub fn candidate_counts(&self) -> rusqlite::Result<Vec<(String, i64)>> {
        self.store.candidates()
    }

    /// Speculative expansions mined for `acronym`, with occurrence counts.
    pub fn potentials_for(&self, acronym: &str) -> rusqlite::Result<Vec<(String, i64)>> {
        self.store.potentials_for(acronym)
    }
}

/// Mine `text` for phrases whose word-initials spell `acronym` — speculative
/// expansions casually mentioned in the same text (no parens required). Uses a
/// strict window of consecutive words (precise; misses skipped function words),
/// de-duplicated within the text.
fn mine_potentials(text: &str, acronym: &str) -> Vec<String> {
    let target: Vec<char> = acronym.chars().map(|c| c.to_ascii_uppercase()).collect();
    let len = target.len();

    // Words trimmed of edge punctuation; drop any that are left empty.
    let words: Vec<&str> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .collect();
    if words.len() < len {
        return Vec::new();
    }

    let mut found = Vec::new();
    for window in words.windows(len) {
        let initials: Vec<char> = window
            .iter()
            .map(|w| w.chars().next().unwrap().to_ascii_uppercase())
            .collect();
        if initials == target {
            let phrase = window.join(" ").to_lowercase();
            if !found.contains(&phrase) {
                found.push(phrase);
            }
        }
    }
    found
}

/// Split text into alphanumeric tokens, preserving each token's original
/// spelling (so `text_slice` reflects what the user wrote).
fn tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
}

/// Acronym-shaped tokens in `text` that were neither expanded nor learned —
/// distinct, in order of first appearance. These are acronyms `ae` *saw* but
/// can't resolve, surfaced so the user can define them.
fn candidate_acronyms(
    text: &str,
    expansions: &[ExpansionResult],
    learned: &[Extraction],
) -> Vec<String> {
    let mut resolved: std::collections::HashSet<String> =
        expansions.iter().map(|e| e.acronym.clone()).collect();
    resolved.extend(learned.iter().map(|c| c.acronym.to_uppercase()));

    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for token in tokens(text) {
        if !is_acronym_shaped(token) {
            continue;
        }
        let upper = token.to_uppercase();
        if !resolved.contains(&upper) && seen.insert(upper) {
            out.push(token.to_string());
        }
    }
    out
}

/// Heuristic acronym shape: 2–6 chars, all uppercase letters or digits, at
/// least one letter, and not starting with a digit (e.g. `MVP`, `S3`, `B2B`).
fn is_acronym_shaped(token: &str) -> bool {
    let len = token.chars().count();
    if !(2..=6).contains(&len) {
        return false;
    }
    let mut has_letter = false;
    for (i, c) in token.chars().enumerate() {
        if c.is_ascii_uppercase() {
            has_letter = true;
        } else if c.is_ascii_digit() {
            if i == 0 {
                return false;
            }
        } else {
            return false;
        }
    }
    has_letter
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
        assert!(first.extractions.iter().any(|c| c.acronym == "ZQ"));
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
    fn read_only_expands_but_learns_nothing() {
        let e = Engine::in_memory().unwrap();
        let out = e.expand_only("The ZQ (Zebra Queue) and the OKR.").unwrap();
        // Seeded OKR still expands; the inline ZQ definition is ignored.
        assert!(out.expansions.iter().any(|r| r.acronym == "OKR"));
        assert!(out.extractions.is_empty());
        // Nothing was persisted: a later pass still doesn't know ZQ.
        let again = e.expand_only("Drain the ZQ now.").unwrap();
        assert!(!again.expansions.iter().any(|r| r.acronym == "ZQ"));
    }

    #[test]
    fn flags_an_unknown_acronym_shaped_token() {
        let e = Engine::in_memory().unwrap();
        let out = e.analyze("hi there MVP and OKR").unwrap();
        // MVP is acronym-shaped but unknown and undefined → flagged.
        assert!(out.candidates.contains(&"MVP".to_string()));
        // OKR is seeded → expanded, not flagged as unknown.
        assert!(out.expansions.iter().any(|r| r.acronym == "OKR"));
        assert!(!out.candidates.contains(&"OKR".to_string()));
    }

    #[test]
    fn an_inline_defined_acronym_is_learned_not_unknown() {
        let e = Engine::in_memory().unwrap();
        let out = e.analyze("see the PDP (Product Detail Page)").unwrap();
        assert!(out.extractions.iter().any(|c| c.acronym == "PDP"));
        assert!(!out.candidates.contains(&"PDP".to_string()));
    }

    #[test]
    fn analyzing_tracks_candidate_frequency() {
        let e = Engine::in_memory().unwrap();
        e.analyze("ship the MVP").unwrap();
        e.analyze("the MVP again, plus XYZ").unwrap();
        let counts = e.candidate_counts().unwrap();
        assert_eq!(
            counts.iter().find(|(a, _)| a == "MVP").map(|(_, n)| *n),
            Some(2)
        );
        assert!(counts.iter().any(|(a, _)| a == "XYZ"));
    }

    #[test]
    fn read_only_does_not_record_candidates() {
        let e = Engine::in_memory().unwrap();
        e.expand_only("ship the MVP").unwrap();
        assert!(e.candidate_counts().unwrap().is_empty());
        assert!(e.potentials_for("MVP").unwrap().is_empty());
    }

    #[test]
    fn mines_speculative_expansions_from_the_same_text() {
        let e = Engine::in_memory().unwrap();
        // No parens — the phrase is just mentioned in the text mentioning MVP.
        e.analyze("the MVP roadmap calls for a minimum viable product first")
            .unwrap();
        let pots = e.potentials_for("MVP").unwrap();
        assert!(pots.iter().any(|(p, _)| p == "minimum viable product"));
    }

    #[test]
    fn speculative_expansions_accrue_with_recurrence() {
        let e = Engine::in_memory().unwrap();
        e.analyze("MVP today means minimum viable product").unwrap();
        e.analyze("our MVP is the minimum viable product").unwrap();
        let pots = e.potentials_for("MVP").unwrap();
        let count = pots
            .iter()
            .find(|(p, _)| p == "minimum viable product")
            .map(|(_, c)| *c);
        assert_eq!(count, Some(2));
    }

    #[test]
    fn defining_an_acronym_clears_its_speculation() {
        let e = Engine::in_memory().unwrap();
        e.analyze("MVP — minimum viable product").unwrap();
        assert!(!e.potentials_for("MVP").unwrap().is_empty());
        e.analyze("MVP (Minimum Viable Product)").unwrap(); // inline definition
        assert!(e.potentials_for("MVP").unwrap().is_empty());
    }

    #[test]
    fn ordinary_lowercase_words_are_not_acronym_candidates() {
        let e = Engine::in_memory().unwrap();
        assert!(
            e.analyze("the cat sat on a mat")
                .unwrap()
                .candidates
                .is_empty()
        );
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
