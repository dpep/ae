//! The hybrid evaluation stack: Stage 1 expansion + Stage 2 learning.
//!
//! An [`Engine`] owns the warm state — the SQLite [`Store`], the in-memory
//! [`SharedTrie`], and an [`Embedder`] — and turns a chunk of text into an
//! [`AnalysisPayload`]. The Leader holds one of these for the process lifetime;
//! the fallback path builds an ephemeral one per invocation.

use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use regex::Regex;

use crate::embed::{Embedder, HashEmbedder};
use crate::mrl::{compress_matryoshka_vector, cosine_similarity};
use crate::store::Store;
use crate::trie::SharedTrie;
use crate::types::{AnalysisPayload, ExpansionResult, MatchCandidate};
use crate::{learn, types::Extraction};

use crate::store::WATCH_THRESHOLD;

/// How often the consolidation auto-job may run, in seconds — `AE_CONSOLIDATE_SECS`,
/// default 1 day. A negative value disables it (used by tests); `0` makes it run
/// every write.
pub fn consolidate_interval_secs() -> Option<i64> {
    let secs: i64 = std::env::var("AE_CONSOLIDATE_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24 * 60 * 60);
    (secs >= 0).then_some(secs)
}

/// Grace period (seconds) before a seen-once candidate is eligible for noise
/// pruning — `AE_PRUNE_GRACE_SECS`, default [`crate::store::DEFAULT_PRUNE_GRACE_SECS`]
/// (~30 days; `0` prunes immediately, for tests). Recently seen tokens are kept
/// so they don't vanish mid-use.
pub fn prune_grace_secs() -> i64 {
    std::env::var("AE_PRUNE_GRACE_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(crate::store::DEFAULT_PRUNE_GRACE_SECS)
}

/// Acronym-shaped tokens with internal punctuation (`PB&J`, `R&D`, `U.S.A`) that
/// the plain alphanumeric tokenizer would split apart.
static PUNCTUATED_ACRONYM: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Z][A-Z0-9]*(?:[&.][A-Z0-9]+)+").unwrap());

/// Backstop so the cached mining trie can't outlive an out-of-band edit that
/// happened to leave the signature unchanged (a rare `rm` + `add` pair).
const MINING_TRIE_MAX_AGE: Duration = Duration::from_secs(300);

/// The cached base mining trie (watch list ∪ known acronyms), with the cheap
/// `(known, watch-list)` signature it was built from and when.
struct MiningCache {
    trie: Arc<MiningTrie>,
    signature: (i64, i64),
    built_at: Instant,
}

pub struct Engine {
    store: Store,
    trie: SharedTrie,
    embedder: Box<dyn Embedder>,
    /// Rebuilt only when the signature changes (or it ages out) — cheap to reuse
    /// across the many requests a warm daemon serves.
    mining_cache: Mutex<Option<MiningCache>>,
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
            mining_cache: Mutex::new(None),
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

    /// Consolidate speculation now — spell-correct + dedup (quality) then drop
    /// low-confidence/noise (cleanup). See [`Store::consolidate`].
    pub fn consolidate(
        &self,
        min_confidence: f32,
        grace_secs: i64,
    ) -> rusqlite::Result<crate::store::ConsolidateStats> {
        self.store.consolidate(min_confidence, grace_secs)
    }

    /// Run consolidation only if the cadence is due (see
    /// [`consolidate_interval_secs`]) — the amortized auto-job after a write.
    pub fn consolidate_if_due(&self, min_confidence: f32, grace_secs: i64) -> rusqlite::Result<()> {
        if let Some(interval) = consolidate_interval_secs()
            && self.store.consolidate_due(interval)?
        {
            self.store.consolidate(min_confidence, grace_secs)?;
        }
        Ok(())
    }

    /// Record candidate sightings, then mine speculative expansions in a single
    /// pass and accumulate vector coherence.
    ///
    /// Mining walks the text once over a [`MiningTrie`], emitting every acronym
    /// whose word initials it spells — O(text), not O(acronyms × text). The base
    /// trie (watch list ∪ known acronyms) is cached across requests (see
    /// [`Self::base_mining_trie`]); the candidates *seen in this text* are mined
    /// via a tiny per-request trie, so a brand-new one is caught on first sight.
    fn mine(&self, text: &str, query_vec: &[f32], candidates: &[String]) -> rusqlite::Result<()> {
        for acronym in candidates {
            self.store.record_candidate(acronym)?;
        }

        let base = self.base_mining_trie()?;
        let mut present = MiningTrie::default();
        for acronym in candidates {
            present.insert(acronym);
        }
        let mut matches: std::collections::HashSet<(String, String)> =
            base.mine(text).into_iter().collect();
        matches.extend(present.mine(text));

        // Route each match: a recurrence of a *known* expansion strengthens its
        // context; anything else is a (new or already-speculative) alternative.
        for (acronym, phrase) in matches {
            let confirmed = self.store.expansions_for(&acronym)?;
            match confirmed
                .iter()
                .find(|(_, e, _)| e.to_lowercase() == phrase)
            {
                Some((id, _, _)) => self.store.add_context(*id, query_vec)?,
                None => {
                    let coherence = self.context_coherence(&acronym, query_vec)?;
                    self.store.record_potential(&acronym, &phrase, coherence)?;
                }
            }
        }

        // Fold this text into each present candidate's context — after mining, so
        // the coherence above reflects prior sightings, not this one.
        for acronym in candidates {
            self.store.update_candidate_context(acronym, query_vec)?;
        }
        Ok(())
    }

    /// The cached base mining trie (watch list ∪ known acronyms, length ≥ 3).
    /// Rebuilt only when the cheap `(known, watch-list)` signature changes —
    /// which also catches out-of-band edits (`add`/`rm`/`watch`/…) since they
    /// move those counts — or when it ages past [`MINING_TRIE_MAX_AGE`].
    fn base_mining_trie(&self) -> rusqlite::Result<Arc<MiningTrie>> {
        let signature = (
            self.store.count()?,
            self.store.watch_list_count(WATCH_THRESHOLD)?,
        );
        let mut cache = self.mining_cache.lock().unwrap();
        let fresh = cache.as_ref().is_some_and(|c| {
            c.signature == signature && c.built_at.elapsed() < MINING_TRIE_MAX_AGE
        });
        if !fresh {
            let mut trie = MiningTrie::default();
            for acronym in self.store.watch_list(WATCH_THRESHOLD)? {
                if acronym.chars().count() >= 3 {
                    trie.insert(&acronym);
                }
            }
            for acronym in self.store.all_acronyms()? {
                if acronym.chars().count() >= 3 {
                    trie.insert(&acronym);
                }
            }
            *cache = Some(MiningCache {
                trie: Arc::new(trie),
                signature,
                built_at: Instant::now(),
            });
        }
        Ok(cache.as_ref().unwrap().trie.clone())
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

/// How many words past the anchor a single match may span — bounds how many
/// fillers we'll skip while spelling out an acronym.
const MAX_MINE_SPAN: usize = 12;

/// A trie of acronyms keyed by their letters (punctuation stripped, so `PB&J`
/// keys as `PBJ`), used to mine a whole text for *every* stored acronym's
/// expansions in one pass instead of rescanning per acronym.
#[derive(Default)]
struct MiningTrie {
    children: std::collections::HashMap<char, MiningTrie>,
    /// Original acronyms whose letters end at this node (e.g. `PB&J`).
    terminals: Vec<String>,
}

impl MiningTrie {
    /// Add `acronym` keyed by its uppercase alphanumeric letters.
    fn insert(&mut self, acronym: &str) {
        let mut node = self;
        let mut any = false;
        for c in acronym
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_uppercase())
        {
            node = node.children.entry(c).or_default();
            any = true;
        }
        if any && !node.terminals.iter().any(|a| a == acronym) {
            node.terminals.push(acronym.to_string());
        }
    }

    /// Every `(acronym, phrase)` the text spells: word-initial subsequences that
    /// reach a terminal, anchored on a content word, tolerating skipped fillers
    /// (and consuming one when it supplies the next letter), with every content
    /// word contributing. De-duplicated.
    fn mine(&self, text: &str) -> Vec<(String, String)> {
        let words: Vec<&str> = text
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();
        let mut found = std::collections::HashSet::new();
        for i in 0..words.len() {
            if !is_filler(words[i]) {
                self.walk(&words, i, i, &mut found);
            }
        }
        found.into_iter().collect()
    }

    /// Walk from this node, consuming `words[j]`. Branches at a filler that
    /// matches an edge: try both consuming it and skipping it.
    fn walk(
        &self,
        words: &[&str],
        anchor: usize,
        j: usize,
        found: &mut std::collections::HashSet<(String, String)>,
    ) {
        if j >= words.len() || j - anchor >= MAX_MINE_SPAN {
            return;
        }
        let init = words[j].chars().next().unwrap().to_ascii_uppercase();
        let filler = is_filler(words[j]);
        if let Some(child) = self.children.get(&init) {
            let phrase = words[anchor..=j].join(" ").to_lowercase();
            for acr in &child.terminals {
                found.insert((acr.clone(), phrase.clone()));
            }
            child.walk(words, anchor, j + 1, found);
            if filler {
                self.walk(words, anchor, j + 1, found); // also try skipping it
            }
        } else if filler {
            self.walk(words, anchor, j + 1, found); // filler that doesn't fit
        }
        // a content word with no matching edge ends this path
    }
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
    fn mining_trie_finds_all_acronyms_in_one_pass() {
        // One traversal yields matches for every stored acronym the text spells.
        let mut t = MiningTrie::default();
        for a in ["OKR", "KPI", "POC", "PB&J"] {
            t.insert(a);
        }
        let hits = t.mine("our objectives and key results, the key performance index, point of contact, and peanut butter and jelly");
        let has = |acr: &str, phrase: &str| hits.iter().any(|(a, p)| a == acr && p == phrase);
        assert!(has("OKR", "objectives and key results"));
        assert!(has("KPI", "key performance index"));
        assert!(has("POC", "point of contact")); // filler "of" consumed
        assert!(has("PB&J", "peanut butter and jelly")); // '&' keyed as PBJ
    }

    #[test]
    fn mining_trie_ignores_a_non_contributing_content_word() {
        // Precision guard: an unrelated content word between letters breaks it.
        let mut t = MiningTrie::default();
        t.insert("ABC");
        let hits = t.mine("apple banana zebra cat");
        assert!(hits.iter().all(|(_, p)| p != "apple banana zebra cat"));
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
    fn consolidate_dedups_and_prunes() {
        let e = Engine::in_memory().unwrap();
        e.declare_acronym("MVP").unwrap();
        // Cross-text mining (MVP declared) records two near-duplicate phrases.
        e.analyze("we want a minimum viable product").unwrap();
        e.analyze("ship a min viable product too").unwrap();
        assert_eq!(e.potentials_for("MVP").unwrap().len(), 2);
        e.consolidate(0.0, 0).unwrap(); // min_confidence 0 → only dedup, no drops
        let pots = e.potentials_for("MVP").unwrap();
        assert_eq!(pots.len(), 1);
        assert!(pots.iter().any(|(p, _)| p == "minimum viable product"));
    }

    #[test]
    fn consolidate_spares_recently_mined_low_confidence_expansions() {
        let e = Engine::in_memory().unwrap();
        e.declare_acronym("MVP").unwrap();
        e.analyze("we want a minimum viable product").unwrap();
        assert!(!e.potentials_for("MVP").unwrap().is_empty());
        // A high floor would drop this speculative row — but it was just mined,
        // so the grace window spares it...
        e.consolidate(0.9, 3600).unwrap();
        assert!(!e.potentials_for("MVP").unwrap().is_empty());
        // ...and with no grace it's dropped.
        e.consolidate(0.9, 0).unwrap();
        assert!(e.potentials_for("MVP").unwrap().is_empty());
    }

    #[test]
    fn known_acronyms_mine_alternative_expansions() {
        let e = Engine::in_memory().unwrap();
        // KPI is a known default ("Key Performance Indicator"). A text whose
        // initials spell KPI with a *different* phrase mines an alternative.
        e.analyze("the kangaroo population index rose").unwrap();
        let pots = e.potentials_for("KPI").unwrap();
        assert!(pots.iter().any(|(p, _)| p == "kangaroo population index"));
    }

    #[test]
    fn mining_trie_cache_picks_up_a_newly_declared_acronym() {
        let e = Engine::in_memory().unwrap();
        // First analysis builds the cache without ZZQ on the watch list.
        e.analyze("nothing to see here").unwrap();
        // Declaring it shifts the watch-list signature...
        e.declare_acronym("ZZQ").unwrap();
        // ...so the next analysis rebuilds the cache and mines it.
        e.analyze("the zebra zoo quarterly opened").unwrap();
        let pots = e.potentials_for("ZZQ").unwrap();
        assert!(pots.iter().any(|(p, _)| p == "zebra zoo quarterly"));
    }

    #[test]
    fn known_expansion_recurrence_is_not_a_duplicate_suggestion() {
        let e = Engine::in_memory().unwrap();
        // Restating OKR's known expansion shouldn't add it as a speculative row.
        e.analyze("our objectives and key results review").unwrap();
        assert!(
            e.potentials_for("OKR")
                .unwrap()
                .iter()
                .all(|(p, _)| p != "objectives and key results")
        );
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
