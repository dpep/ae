//! The hybrid evaluation stack: Stage 1 expansion + Stage 2 learning.
//!
//! An [`Engine`] owns the warm state — the SQLite [`Store`], the in-memory
//! [`SharedTrie`], and an [`Embedder`] — and turns a chunk of text into an
//! [`AnalysisPayload`]. The Leader holds one of these for the process lifetime;
//! the fallback path builds an ephemeral one per invocation.

use std::sync::LazyLock;

use regex::Regex;

use crate::embed::{Embedder, HashEmbedder};
use crate::mrl::{compress_matryoshka_vector, cosine_similarity};
use crate::store::Store;
use crate::trie::SharedTrie;
use crate::types::{AnalysisPayload, ExpansionResult, MatchCandidate};
use crate::{learn, types::Extraction};

use crate::store::WATCH_THRESHOLD;

/// Whether to run the amortized GC after a write — a small random chance
/// (`AE_GC_PERCENT`, default 5; `0` disables, used by tests). Cheap entropy from
/// the clock; this is sampling, not security.
pub fn should_gc() -> bool {
    let percent: u32 = std::env::var("AE_GC_PERCENT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    if percent == 0 {
        return false;
    }
    if percent >= 100 {
        return true;
    }
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    (nanos % 100) < percent
}

/// Grace period (seconds) before a seen-once candidate is eligible for noise
/// pruning — `AE_PRUNE_GRACE_SECS`, default 1 hour (`0` prunes immediately, for
/// tests). Recently seen tokens are kept so they don't vanish mid-use.
pub fn prune_grace_secs() -> i64 {
    std::env::var("AE_PRUNE_GRACE_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3600)
}

/// Acronym-shaped tokens with internal punctuation (`PB&J`, `R&D`, `U.S.A`) that
/// the plain alphanumeric tokenizer would split apart.
static PUNCTUATED_ACRONYM: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Z][A-Z0-9]*(?:[&.][A-Z0-9]+)+").unwrap());

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

        // Stage 3: candidate tracking + speculative expansion mining (analysis
        // only — read-only leaves no trace).
        self.mine(text, &query_vec, &candidates)?;

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
        // Maximal munch: don't expand a "PB" that's really part of a "PB&J".
        let covered = covered_parts(text);

        for token in punctuated_acronyms(text).chain(tokens(text)) {
            if !self.trie.contains(token) {
                continue;
            }
            let acronym = token.to_uppercase();
            if covered.contains(&acronym) || results.iter().any(|r| r.acronym == acronym) {
                continue; // a longer acronym covers it, or already handled
            }

            let mut matches: Vec<MatchCandidate> = self
                .store
                .expansions_for(&acronym)?
                .into_iter()
                .map(|(id, expansion, source)| {
                    Ok(MatchCandidate {
                        expansion,
                        validity: crate::store::source_validity(&source),
                        confidence: self.contextual(id, query_vec)?,
                    })
                })
                .collect::<rusqlite::Result<_>>()?;

            // Best fit for *this* context first, then most-valid.
            matches.sort_by(|a, b| {
                b.confidence
                    .total_cmp(&a.confidence)
                    .then(b.validity.total_cmp(&a.validity))
            });

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

    /// Contextual likelihood for one expansion: how well the query sentence
    /// matches the expansion's recorded contexts (`0.5` — neutral — with no
    /// evidence yet). This is P(expansion | acronym, context), distinct from
    /// validity.
    fn contextual(&self, id: i64, query_vec: &[f32]) -> rusqlite::Result<f32> {
        let contexts = self.store.contexts_for(id)?;
        if contexts.is_empty() {
            return Ok(0.5);
        }
        let best = contexts
            .iter()
            .map(|c| cosine_similarity(query_vec, c))
            .fold(0.0_f32, f32::max)
            .clamp(0.0, 1.0);
        Ok(best)
    }

    /// Stage 2 — extract inline definitions, persist them (dictionary + trie +
    /// a context embedding), and return them.
    fn learn_and_persist(&self, text: &str) -> rusqlite::Result<Vec<Extraction>> {
        let learned = learn::extract(text);
        for c in &learned {
            let id = self
                .store
                .add_entry(&c.acronym, &c.extracted_definition, "inline")?;
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

    /// Declare a token is an acronym (user provenance) — see
    /// [`Store::declare_acronym`].
    pub fn declare_acronym(&self, acronym: &str) -> rusqlite::Result<()> {
        self.store.declare_acronym(acronym)
    }

    /// Speculative expansions mined for `acronym`, with occurrence counts.
    pub fn potentials_for(&self, acronym: &str) -> rusqlite::Result<Vec<(String, i64)>> {
        Ok(self
            .store
            .potentials_for(acronym)?
            .into_iter()
            .map(|(expansion, count, _coh)| (expansion, count))
            .collect())
    }

    /// Amortized GC: dedup mined expansions, drop low-confidence ones, and prune
    /// noise candidates. The cheap subset of `ae prune` (no spell-correction),
    /// run occasionally after a write to spread the cost — see [`should_gc`].
    pub fn gc(&self, min_confidence: f32) -> rusqlite::Result<()> {
        for acronym in self.store.distinct_potential_acronyms()? {
            self.store.dedup_potentials(&acronym)?;
        }
        let all = self.store.all_potentials()?;
        let mut totals: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        for (acronym, _, count, _) in &all {
            *totals.entry(acronym.clone()).or_insert(0) += count;
        }
        for (acronym, expansion, count, coh) in all {
            if crate::store::confidence(count, coh, totals[&acronym]) < min_confidence {
                self.store.delete_potential(&acronym, &expansion)?;
            }
        }
        self.store.prune_noise_candidates(prune_grace_secs())?;
        Ok(())
    }

    /// Record candidates, mine speculative expansions (same-text and, for other
    /// watched candidates, cross-text), and accumulate vector coherence.
    fn mine(&self, text: &str, query_vec: &[f32], candidates: &[String]) -> rusqlite::Result<()> {
        let present: std::collections::HashSet<String> =
            candidates.iter().map(|c| c.to_uppercase()).collect();

        // Candidates mentioned in this text: count them, mine here, and fold
        // this text into the acronym's context (where it tends to appear).
        for acronym in candidates {
            self.store.record_candidate(acronym)?;
            let coherence = self.context_coherence(acronym, query_vec)?;
            for phrase in mine_potentials(text, acronym) {
                self.store.record_potential(acronym, &phrase, coherence)?;
            }
            self.store.update_candidate_context(acronym, query_vec)?;
        }

        // Cross-text lookout: scan this text for the expansions of *other*
        // watch-list acronyms — declared, or seen often enough to be promoted
        // from noise to mining (length ≥ 3 limits short-acronym noise).
        for acronym in self.store.watch_list(WATCH_THRESHOLD)? {
            if present.contains(&acronym) || acronym.chars().count() < 3 {
                continue;
            }
            let coherence = self.context_coherence(&acronym, query_vec)?;
            for phrase in mine_potentials(text, &acronym) {
                self.store.record_potential(&acronym, &phrase, coherence)?;
            }
        }
        Ok(())
    }

    /// Cosine of this text's embedding against where `acronym` usually appears
    /// (`1.0` when we have no history yet — neutral, don't penalize).
    fn context_coherence(&self, acronym: &str, query_vec: &[f32]) -> rusqlite::Result<f32> {
        Ok(match self.store.candidate_context_mean(acronym)? {
            Some(mean) => cosine_similarity(query_vec, &mean).clamp(0.0, 1.0),
            None => 1.0,
        })
    }
}

/// Short function words an acronym may omit (e.g. OKR = Objectives *and* Key
/// Results). A filler is *skipped* when it doesn't help, but still *consumed*
/// when its initial does match the next letter (e.g. POC = Point *of* Contact).
const FILLER: &[&str] = &[
    "a", "an", "and", "the", "of", "for", "to", "in", "on", "at", "by", "with", "or", "as", "per",
];

fn is_filler(word: &str) -> bool {
    FILLER.contains(&word.to_lowercase().as_str())
}

/// Mine `text` for phrases whose word-initials spell `acronym` — speculative
/// expansions casually mentioned (no parens required). Matches the acronym as a
/// subsequence over words, tolerating skipped filler words, but every *content*
/// word must contribute a letter (the precision guard) and the phrase is
/// anchored at content words on both ends. De-duplicated within the text.
fn mine_potentials(text: &str, acronym: &str) -> Vec<String> {
    // Letters only — punctuation in the acronym (PB&J, R&D) isn't a word
    // initial; the '&'/dot maps to a filler word ("and") we already skip.
    let target: Vec<char> = acronym
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .collect();
    if target.is_empty() {
        return Vec::new();
    }
    let words: Vec<&str> = text
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .collect();
    if words.len() < target.len() {
        return Vec::new();
    }
    let initial = |w: &str| w.chars().next().unwrap().to_ascii_uppercase();
    let max_span = target.len() * 2 + 2; // bound how many fillers we'll skip

    let mut found = Vec::new();
    for i in 0..words.len() {
        // Anchor on a content word that opens the acronym.
        if is_filler(words[i]) || initial(words[i]) != target[0] {
            continue;
        }
        let (mut t, mut last, mut j) = (1usize, i, i + 1);
        while j < words.len() && t < target.len() && j - i < max_span {
            if initial(words[j]) == target[t] {
                t += 1;
                last = j;
            } else if !is_filler(words[j]) {
                break; // a content word that doesn't fit ends the window
            }
            j += 1;
        }
        if t == target.len() {
            let phrase = words[i..=last].join(" ").to_lowercase();
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

/// Punctuated acronym mentions (`PB&J`, `R&D`) — 2–6 letters, all uppercase /
/// digit / `&` / `.`, with at least one of `&`/`.`.
fn punctuated_acronyms(text: &str) -> impl Iterator<Item = &str> {
    PUNCTUATED_ACRONYM
        .find_iter(text)
        .map(|m| m.as_str())
        .filter(|t| {
            let letters = t.chars().filter(|c| c.is_ascii_uppercase()).count();
            (2..=6).contains(&letters)
        })
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

    // Punctuated acronyms first (maximal munch): a "PB&J" suppresses its parts
    // "PB" and "J".
    let covered = covered_parts(text);
    for token in punctuated_acronyms(text) {
        let upper = token.to_uppercase();
        if !resolved.contains(&upper) && seen.insert(upper) {
            out.push(token.to_string());
        }
    }
    for token in tokens(text) {
        let upper = token.to_uppercase();
        if !is_acronym_shaped(token) || covered.contains(&upper) {
            continue;
        }
        if !resolved.contains(&upper) && seen.insert(upper) {
            out.push(token.to_string());
        }
    }
    out
}

/// Uppercase sub-tokens covered by a punctuated acronym (`PB&J` → `{PB, J}`),
/// so the longer match suppresses its parts (maximal munch).
fn covered_parts(text: &str) -> std::collections::HashSet<String> {
    let mut covered = std::collections::HashSet::new();
    for token in punctuated_acronyms(text) {
        for part in token.split(|c: char| !c.is_alphanumeric()) {
            if !part.is_empty() {
                covered.insert(part.to_uppercase());
            }
        }
    }
    covered
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
    fn mining_skips_a_filler_word_that_is_not_in_the_acronym() {
        let e = Engine::in_memory().unwrap();
        // PBJ skips "and": Peanut Butter (and) Jelly.
        e.analyze("a classic PBJ is peanut butter and jelly")
            .unwrap();
        let pots = e.potentials_for("PBJ").unwrap();
        assert!(pots.iter().any(|(p, _)| p == "peanut butter and jelly"));
    }

    #[test]
    fn mining_consumes_a_filler_word_that_supplies_a_letter() {
        let e = Engine::in_memory().unwrap();
        // POC uses the "of": Point Of Contact.
        e.analyze("our POC is the point of contact for vendors")
            .unwrap();
        let pots = e.potentials_for("POC").unwrap();
        assert!(pots.iter().any(|(p, _)| p == "point of contact"));
    }

    #[test]
    fn cross_text_mining_waits_for_the_watch_threshold() {
        let e = Engine::in_memory().unwrap();
        // Seen twice (< threshold of 3) → not yet on the watch list.
        e.analyze("the MVP ships next week").unwrap();
        e.analyze("ping me about the MVP today").unwrap();
        e.analyze("we scoped a minimum viable product for launch")
            .unwrap();
        assert!(e.potentials_for("MVP").unwrap().is_empty());

        // A third sighting promotes it; now cross-text mining picks the phrase up.
        e.analyze("the MVP demo is friday").unwrap();
        e.analyze("we shipped a minimum viable product").unwrap();
        let pots = e.potentials_for("MVP").unwrap();
        assert!(pots.iter().any(|(p, _)| p == "minimum viable product"));
    }

    #[test]
    fn declaring_an_acronym_makes_it_mine_worthy_immediately() {
        let e = Engine::in_memory().unwrap();
        e.declare_acronym("MVP").unwrap();
        // Never seen as a token, but a phrase in this text is mined for it.
        e.analyze("we scoped a minimum viable product for launch")
            .unwrap();
        let pots = e.potentials_for("MVP").unwrap();
        assert!(pots.iter().any(|(p, _)| p == "minimum viable product"));
    }

    #[test]
    fn gc_dedups_and_prunes() {
        let e = Engine::in_memory().unwrap();
        e.declare_acronym("MVP").unwrap();
        // Cross-text mining (MVP declared) records two near-duplicate phrases.
        e.analyze("we want a minimum viable product").unwrap();
        e.analyze("ship a min viable product too").unwrap();
        assert_eq!(e.potentials_for("MVP").unwrap().len(), 2);
        e.gc(0.0).unwrap(); // min_confidence 0 → only dedup, no drops
        let pots = e.potentials_for("MVP").unwrap();
        assert_eq!(pots.len(), 1);
        assert!(pots.iter().any(|(p, _)| p == "minimum viable product"));
    }

    #[test]
    fn punctuated_acronyms_are_detected_and_mined() {
        let e = Engine::in_memory().unwrap();
        let out = e.analyze("a PB&J is peanut butter and jelly").unwrap();
        assert!(out.candidates.contains(&"PB&J".to_string()));
        let pots = e.potentials_for("PB&J").unwrap();
        assert!(pots.iter().any(|(p, _)| p == "peanut butter and jelly"));
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
