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
/// (7 days; `0` prunes immediately, for tests). Recently seen tokens are kept
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
/// store signature it — and the expansion trie — were last synced at, and when.
struct MiningCache {
    trie: Arc<MiningTrie>,
    signature: (i64, i64, i64),
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

    /// Which embedder backs this engine: `"onnx"` (real model) or `"hash"`
    /// (deterministic fallback). Surfaced by `ae --status`.
    pub fn embedder_kind(&self) -> &'static str {
        self.embedder.kind()
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
        let base = self.sync_dictionary()?;
        let query_vec = QueryVec::new(self.embedder.as_ref(), text);
        let ignored = self.store.ignored_set()?;

        let expansions = self.expand(text, &query_vec, &ignored)?;
        let (confirmed, speculative) = self.learn_and_persist(text, &ignored)?;
        // An unconfirmed acronym is still undefined, so it stays a candidate.
        let candidates = candidate_acronyms(text, &expansions, &confirmed, &ignored);

        // Stage 3: candidate tracking + speculative expansion mining (analysis
        // only — read-only leaves no trace).
        self.mine(text, &query_vec, &candidates, &base)?;

        Ok(AnalysisPayload {
            sentence: text.to_string(),
            expansions,
            extractions: confirmed.into_iter().chain(speculative).collect(),
            candidates,
        })
    }

    /// Read-only Stage 1 only: expand known acronyms (and flag unknown ones)
    /// without extracting or persisting anything. The dictionary is never
    /// written, so this is safe to run against a shared DB without side effects.
    pub fn expand_only(&self, text: &str) -> rusqlite::Result<AnalysisPayload> {
        self.sync_dictionary()?;
        let query_vec = QueryVec::new(self.embedder.as_ref(), text);
        let ignored = self.store.ignored_set()?;
        let expansions = self.expand(text, &query_vec, &ignored)?;
        let candidates = candidate_acronyms(text, &expansions, &[], &ignored);
        Ok(AnalysisPayload {
            sentence: text.to_string(),
            expansions,
            extractions: Vec::new(),
            candidates,
        })
    }

    /// Stage 1 — scan the text for known acronyms and rank their expansions.
    /// `ignored` acronyms are skipped even if the (possibly warm) trie still
    /// holds them, so muting takes effect without rebuilding the trie.
    fn expand(
        &self,
        text: &str,
        query_vec: &QueryVec,
        ignored: &std::collections::HashSet<String>,
    ) -> rusqlite::Result<Vec<ExpansionResult>> {
        let mut results: Vec<ExpansionResult> = Vec::new();
        // Maximal munch: don't expand a "PB" that's really part of a "PB&J".
        let covered = covered_parts(text);

        for token in punctuated_acronyms(text).chain(tokens(text)) {
            if !self.trie.contains(token) {
                continue;
            }
            let acronym = token.to_uppercase();
            if ignored.contains(&acronym)
                || covered.contains(&acronym)
                || results.iter().any(|r| r.acronym == acronym)
            {
                continue; // muted, a longer acronym covers it, or already handled
            }

            let entries = self.store.expansions_for(&acronym)?;
            // A sole expansion has nothing to disambiguate — skip the inference.
            let fits = if entries.len() > 1 {
                entries
                    .iter()
                    .map(|(id, expansion, _)| self.fit(*id, expansion, query_vec))
                    .collect::<rusqlite::Result<Vec<_>>>()?
            } else {
                vec![0.0; entries.len()]
            };
            let mut matches: Vec<MatchCandidate> = entries
                .into_iter()
                .zip(fit_shares(&fits))
                .map(|((_, expansion, source), share)| MatchCandidate {
                    expansion,
                    validity: crate::store::source_validity(&source),
                    confidence: share,
                    source,
                })
                .collect();

            // Rank by the exposed score (validity × share), validity breaking ties.
            matches.sort_by(|a, b| {
                (b.validity * b.confidence)
                    .total_cmp(&(a.validity * a.confidence))
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

    /// Raw contextual fit for one expansion: the best cosine between the query
    /// and its recorded contexts. With none recorded, it's scored against its
    /// own phrase — the very vector its first context would hold — so a
    /// verified entry nobody has seen in the wild isn't penalized against a
    /// mined sibling. Only meaningful relative to siblings (see [`fit_shares`]).
    fn fit(&self, id: i64, expansion: &str, query_vec: &QueryVec) -> rusqlite::Result<f32> {
        let contexts = self.store.contexts_for(id)?;
        if contexts.is_empty() {
            let own = compress_matryoshka_vector(&self.embedder.embed(expansion));
            return Ok(cosine_similarity(query_vec.get(), &own));
        }
        Ok(contexts
            .iter()
            .map(|c| cosine_similarity(query_vec.get(), c))
            .fold(f32::NEG_INFINITY, f32::max))
    }

    /// Stage 2 — extract inline definitions and return them as `(confirmed,
    /// speculative)`. Aligned ones (initials spell the acronym) are confirmed:
    /// persisted as `inline` with a context embedding. The rest are mostly
    /// fragments and names, so they only accrue as speculation for `suggest`,
    /// where recurrence has to earn them a place.
    fn learn_and_persist(
        &self,
        text: &str,
        ignored: &std::collections::HashSet<String>,
    ) -> rusqlite::Result<(Vec<Extraction>, Vec<Extraction>)> {
        // A muted acronym stays dormant — don't persist, expand, or surface its
        // inline definition.
        let (confirmed, speculative): (Vec<Extraction>, Vec<Extraction>) = learn::extract(text)
            .into_iter()
            .filter(|c| !ignored.contains(&c.acronym.to_uppercase()))
            .partition(learn::is_aligned);
        for c in &confirmed {
            let id = self
                .store
                .add_entry(&c.acronym, &c.extracted_definition, "inline")?;
            self.trie.insert(&c.acronym);
            let ctx = compress_matryoshka_vector(&self.embedder.embed(&c.extracted_definition));
            self.store.add_context(id, &ctx)?;
        }
        for c in &speculative {
            // The author paired them in this very text: same-sentence evidence.
            self.store.record_potential(
                &c.acronym,
                &c.extracted_definition,
                SAME_SENTENCE_COHERENCE,
            )?;
        }
        Ok((confirmed, speculative))
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

    /// Mute an acronym — see [`Store::ignore_acronym`].
    pub fn ignore_acronym(&self, acronym: &str) -> rusqlite::Result<()> {
        self.store.ignore_acronym(acronym)
    }

    /// Un-mute an acronym — see [`Store::unignore_acronym`].
    pub fn unignore_acronym(&self, acronym: &str) -> rusqlite::Result<bool> {
        self.store.unignore_acronym(acronym)
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
    /// whose word initials it spells — O(text), not O(acronyms × text). The
    /// `base` trie (watch list ∪ known acronyms) is cached across requests (see
    /// [`Self::sync_dictionary`]); the candidates *seen in this text* are mined
    /// via a tiny per-request trie, so a brand-new one is caught on first sight.
    fn mine(
        &self,
        text: &str,
        query_vec: &QueryVec,
        candidates: &[String],
        base: &MiningTrie,
    ) -> rusqlite::Result<()> {
        for acronym in candidates {
            self.store.record_candidate(acronym)?;
        }

        let mut present = MiningTrie::default();
        for acronym in candidates.iter().filter(|a| is_mine_worthy(a)) {
            present.insert(acronym);
        }
        // Present-trie hits are *same-sentence* co-occurrences: the acronym
        // appeared as a candidate in this very text alongside the phrase that
        // spells it. That's first-hand evidence, stronger than a cross-text
        // (base-trie) guess, so it earns full coherence credit.
        let present_matches: std::collections::HashSet<(String, String)> =
            present.mine(text).into_iter().collect();
        let mut matches: std::collections::HashSet<(String, String)> =
            base.mine(text).into_iter().collect();
        matches.extend(present_matches.iter().cloned());

        // Route each match: a recurrence of a *known* expansion strengthens its
        // context; anything else is a (new or already-speculative) alternative.
        for (acronym, phrase) in matches {
            let confirmed = self.store.expansions_for(&acronym)?;
            match confirmed
                .iter()
                .find(|(_, e, _)| e.to_lowercase() == phrase)
            {
                // Store the matched phrase's embedding, not the whole-sentence
                // query_vec: scoring a later identical sentence against a stored
                // copy of itself self-matches to cosine 1.0 (and makes the warm
                // daemon, which accumulates that vector, disagree with a fresh
                // in-process run). Mirrors learn_and_persist storing embed(def).
                Some((id, _, _)) => {
                    let ctx = compress_matryoshka_vector(&self.embedder.embed(&phrase));
                    self.store.add_context(*id, &ctx)?;
                }
                None => {
                    let coherence = if present_matches.contains(&(acronym.clone(), phrase.clone()))
                    {
                        SAME_SENTENCE_COHERENCE
                    } else {
                        self.context_coherence(&acronym, query_vec)?
                    };
                    self.store.record_potential(&acronym, &phrase, coherence)?;
                }
            }
        }

        // Fold this text into each present candidate's context — after mining, so
        // the coherence above reflects prior sightings, not this one.
        for acronym in candidates {
            self.store
                .update_candidate_context(acronym, query_vec.get())?;
        }
        Ok(())
    }

    /// Bring the warm tries in line with the store and return the base mining
    /// trie (watch list ∪ known acronyms, 3+ letters). Both are rebuilt only
    /// when the cheap `(known, watch-list, ignored)` signature changes — which
    /// catches out-of-band edits from another process (`add`/`rm`/`watch`/…)
    /// since they move those counts — or when it ages past
    /// [`MINING_TRIE_MAX_AGE`].
    fn sync_dictionary(&self) -> rusqlite::Result<Arc<MiningTrie>> {
        let signature = (
            self.store.count()?,
            self.store.watch_list_count()?,
            self.store.ignored_count()?,
        );
        let mut cache = self.mining_cache.lock().unwrap();
        if let Some(c) = cache
            .as_ref()
            .filter(|c| c.signature == signature && c.built_at.elapsed() < MINING_TRIE_MAX_AGE)
        {
            return Ok(c.trie.clone());
        }
        let known = self.store.all_acronyms()?;
        self.trie.replace(&known);
        let watched = self.store.watch_list()?;
        let mut trie = MiningTrie::default();
        for acronym in watched.iter().chain(&known).filter(|a| is_mine_worthy(a)) {
            trie.insert(acronym);
        }
        let trie = Arc::new(trie);
        *cache = Some(MiningCache {
            trie: trie.clone(),
            signature,
            built_at: Instant::now(),
        });
        Ok(trie)
    }

    /// Cosine of this text's embedding against where `acronym` usually appears
    /// (`1.0` when we have no history yet — neutral, don't penalize).
    fn context_coherence(&self, acronym: &str, query_vec: &QueryVec) -> rusqlite::Result<f32> {
        Ok(match self.store.candidate_context_mean(acronym)? {
            Some(mean) => cosine_similarity(query_vec.get(), &mean).clamp(0.0, 1.0),
            None => 1.0,
        })
    }
}

/// Softmax temperature over sibling fits, in cosine units: a sibling that fits
/// the context 0.1 better is e (≈2.7×) times as likely. Raw cosines from a
/// small embedding model sit in a narrow band, so they can't be read as
/// probabilities directly; this is the scale that turns their gaps into odds.
const FIT_TEMPERATURE: f32 = 0.1;

/// P(expansion | acronym, context) for each sibling: a softmax over their raw
/// fits, so the shares sum to 1. A sole expansion gets 1.0; siblings that fit
/// equally well split evenly.
fn fit_shares(fits: &[f32]) -> Vec<f32> {
    let best = fits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = fits
        .iter()
        .map(|f| ((f - best) / FIT_TEMPERATURE).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

/// Whether an acronym is long enough to mine. Two letters match nearly any
/// adjacent word pair, so their "expansions" are noise. Counts letters, not
/// characters — `C.S` is two letters, not three.
fn is_mine_worthy(acronym: &str) -> bool {
    acronym
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .count()
        >= 3
}

/// The text's embedding, computed on first use. Inference dominates the cost of
/// an analysis, and most lines never need it — nothing to score, no candidate
/// context to fold into.
struct QueryVec<'a> {
    embedder: &'a dyn Embedder,
    text: &'a str,
    vec: std::cell::OnceCell<Vec<f32>>,
}

impl<'a> QueryVec<'a> {
    fn new(embedder: &'a dyn Embedder, text: &'a str) -> Self {
        Self {
            embedder,
            text,
            vec: std::cell::OnceCell::new(),
        }
    }

    fn get(&self) -> &[f32] {
        self.vec
            .get_or_init(|| compress_matryoshka_vector(&self.embedder.embed(self.text)))
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

/// Coherence credited to a *same-sentence* mined expansion — one whose acronym
/// appeared in the very text that spells it out. The co-occurrence is direct
/// evidence, so it gets the maximum (1.0), ranking it above cross-text guesses
/// in `suggest` without being promoted to a confident extraction.
const SAME_SENTENCE_COHERENCE: f32 = 1.0;

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
///
/// An underscore binds rather than separates. Splitting on it turns
/// `MIN_CONFIDENCE` into `MIN` and `CLAUDE_PLUGIN_ROOT` into three tokens that
/// each pass for an acronym, which is how command output floods the candidate
/// list. Kept together, the whole identifier is too long to be acronym-shaped
/// and falls out on its own.
///
/// Leading and trailing underscores are trimmed, so markdown emphasis (`_API_`)
/// still reads as the acronym it wraps — only an underscore *between* two parts
/// glues them.
fn tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .map(|t| t.trim_matches('_'))
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
    ignored: &std::collections::HashSet<String>,
) -> Vec<String> {
    // Casing only marks an acronym when there's lowercase to contrast against.
    // An all-caps line (a shouted headline, an all-caps log) would otherwise
    // flag every short word, so we surface no candidates from it. A lone token
    // is still flagged — there's nothing for it to be "shouting" relative to.
    if is_all_caps_prose(text) {
        return Vec::new();
    }

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
        if ignored.contains(&upper) {
            continue;
        }
        if !resolved.contains(&upper) && seen.insert(upper) {
            out.push(token.to_string());
        }
    }
    // Tokenize per word, so an identifier can be skipped whole. Splitting the
    // full text first would have already shattered `CLAUDE_PLUGIN_ROOT` into
    // three plausible-looking acronyms.
    for word in text.split_whitespace() {
        if is_identifier_word(word) {
            continue;
        }
        for token in tokens(word) {
            let upper = token.to_uppercase();
            if !is_acronym_shaped(token)
                || is_noise(token, crate::spell::system_short_words())
                || covered.contains(&upper)
                || ignored.contains(&upper)
            {
                continue;
            }
            if !resolved.contains(&upper) && seen.insert(upper) {
                out.push(token.to_string());
            }
        }
    }
    out
}

/// True when a whitespace-delimited word is an identifier, path, or shell
/// variable rather than prose — so its uppercase runs are structure, not
/// acronyms.
///
/// Tokenizing splits on punctuation, which turns `$HOME` into `HOME` and
/// `SKILL.md` into `SKILL` — each 2–6 uppercase letters, so each looks exactly
/// like an acronym. Streaming command output past this produces hundreds of
/// candidates that were never words.
///
/// Underscores are handled in [`tokens`] instead, by not splitting on them.
fn is_identifier_word(word: &str) -> bool {
    // $HOME, ${HOME} — a variable reference, not a mention.
    if word.starts_with('$') || word.contains("${") {
        return true;
    }
    // plugins/code/SKILL, C:\Users — a path.
    if word.contains('/') || word.contains('\\') {
        return true;
    }
    // SKILL.md, config.API.url — a dot binding two parts together. A trailing
    // dot is sentence punctuation ("we shipped the MVP.") and doesn't count,
    // which is why this needs the character after it.
    let chars: Vec<char> = word.chars().collect();
    chars.iter().enumerate().any(|(i, c)| {
        *c == '.'
            && chars
                .get(i + 1)
                .is_some_and(|next| next.is_ascii_alphanumeric())
    })
}

/// True when `text` has multiple words and not a single lowercase letter — the
/// all-caps shape where uppercase carries no acronym signal, so we don't mine
/// candidates from it. A single token (no surrounding contrast) is exempt.
fn is_all_caps_prose(text: &str) -> bool {
    let multiword = text.split_whitespace().nth(1).is_some();
    multiword && !text.chars().any(|c| c.is_ascii_lowercase())
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

/// Acronym-shaped tokens that are clearly not acronyms. Applied only to
/// candidates: a dictionary entry is resolved before this runs, so `ae add CAT`
/// still expands.
fn is_noise(token: &str, words: Option<&std::collections::HashSet<String>>) -> bool {
    is_numbered_label(token)
        || NOISE_TOKENS.contains(&token)
        || words.is_some_and(|w| w.contains(&token.to_lowercase()))
}

/// Shouted words the system wordlist lacks (it predates them) — log
/// levels (`INFO`, `DEBUG`) and code annotations (`TODO`, `FIXME`).
const NOISE_TOKENS: &[&str] = &["INFO", "DEBUG", "TODO", "FIXME"];

/// `V1`, `L4`, `Q1`, `H1`, `P0`, `FY27`: a version, level, quarter, half,
/// priority, or fiscal year. No shape tells `Q1` from `S3`, so the prefixes are
/// listed; anything with a letter after its digits (`K8S`, `P2P`) is kept.
fn is_numbered_label(token: &str) -> bool {
    let digits = token.trim_start_matches(|c: char| c.is_ascii_uppercase());
    let prefix = &token[..token.len() - digits.len()];
    !digits.is_empty()
        && digits.bytes().all(|b| b.is_ascii_digit())
        && matches!(prefix, "V" | "L" | "Q" | "H" | "P" | "FY")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts inference calls, so tests can assert when the model is skipped.
    struct CountingEmbedder(Arc<AtomicUsize>);

    impl Embedder for CountingEmbedder {
        fn embed(&self, text: &str) -> Vec<f32> {
            self.0.fetch_add(1, Ordering::Relaxed);
            HashEmbedder::new().embed(text)
        }

        fn kind(&self) -> &'static str {
            "counting"
        }
    }

    fn counting_engine() -> (Engine, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let store = Store::open_in_memory().unwrap();
        store.seed_defaults().unwrap();
        let e = Engine::new(store, Box::new(CountingEmbedder(calls.clone()))).unwrap();
        (e, calls)
    }

    #[test]
    fn two_letter_acronyms_are_not_mined() {
        let e = Engine::in_memory().unwrap();
        for _ in 0..3 {
            e.analyze("the C.S and ZQ teams own the call sites and zone quotas")
                .unwrap();
        }
        assert!(e.potentials_for("C.S").unwrap().is_empty());
        assert!(e.potentials_for("ZQ").unwrap().is_empty());
    }

    #[test]
    fn plain_prose_skips_inference() {
        let (e, calls) = counting_engine();
        e.analyze("nothing much happened here today").unwrap();
        e.expand_only("nothing much happened here today").unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_candidate_still_gets_embedded() {
        let (e, calls) = counting_engine();
        e.analyze("the ZQX rollout slipped").unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

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
    fn repeated_identical_analysis_does_not_self_match() {
        // Regression: mining once stored the whole-sentence vector as a context
        // for a confirmed expansion, so re-analyzing the same sentence scored it
        // against a stored copy of itself → cosine 1.0 (and made a warm daemon
        // disagree with a fresh engine). Confidence must stay stable, not jump
        // to a self-match on the second identical pass. Checked on the raw fit:
        // a sole expansion's share is 1.0 by definition, whatever its contexts.
        let e = Engine::in_memory().unwrap();
        let text = "check TPS (Test Procedure Spec) for the run";
        let fit = || {
            let (id, expansion, _) = e.store.expansions_for("TPS").unwrap().remove(0);
            e.fit(id, &expansion, &QueryVec::new(e.embedder.as_ref(), text))
                .unwrap()
        };
        e.analyze(text).unwrap();
        let f1 = fit();
        e.analyze(text).unwrap();
        e.analyze(text).unwrap();
        let f2 = fit();
        assert!(f2 < 0.99, "self-contaminated fit: {f2}");
        assert!(f2 > 0.0, "a zero query vector would make this vacuous");
        assert!((f1 - f2).abs() < 1e-6, "fit drifted: {f1} vs {f2}");
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
    fn unknown_acronym_fixtures_stay_unseeded() {
        // These acronyms are used across the candidate/mining tests as
        // genuinely-unknown tokens (MVP for its dual meaning + prefix variants,
        // PBJ/POC for filler-word mining). If one ever lands in the seed
        // dictionary it would expand instead of being flagged/mined, silently
        // breaking ~dozens of tests — which is exactly what happened when MVP
        // was briefly seeded. Guard against it.
        let seeded: std::collections::HashSet<&str> = crate::store::DEFAULT_DICTIONARY
            .iter()
            .map(|(a, _)| *a)
            .collect();
        for token in ["MVP", "XYZ", "PBJ", "POC"] {
            assert!(
                !seeded.contains(token),
                "{token} is seeded — it's used as an unknown fixture in the tests"
            );
        }
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
    fn consolidate_drops_a_seen_once_expansion_past_grace_but_keeps_a_recurring_one() {
        let e = Engine::in_memory().unwrap();
        e.declare_acronym("MVP").unwrap();
        e.analyze("we want a minimum viable product").unwrap();
        e.analyze("ship the minimum viable product").unwrap();
        e.analyze("a marvelous velvet pillow").unwrap();
        // A floor of 0 drops nothing on confidence alone; only recurrence counts.
        e.consolidate(0.0, 0).unwrap();
        let pots = e.potentials_for("MVP").unwrap();
        assert!(pots.iter().any(|(p, _)| p == "minimum viable product"));
        assert!(pots.iter().all(|(p, _)| p != "marvelous velvet pillow"));
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
    fn a_same_sentence_spelling_gets_full_coherence_credit() {
        let e = Engine::in_memory().unwrap();
        e.declare_acronym("PQR").unwrap();
        // Prior, unrelated context — so a cross-text mention would score the
        // current text's coherence below 1.0.
        e.analyze("the PQR project timeline shifted").unwrap();
        // Same sentence: the acronym token co-occurs with a phrase spelling it,
        // so the mined suggestion is credited maximal coherence (1.0). Read the
        // raw coh_sum from the store (the Engine accessor drops it).
        e.analyze("our PQR is the peaceful quiet retreat").unwrap();
        let pots = e.store.potentials_for("PQR").unwrap();
        let (_, count, coh) = pots
            .iter()
            .find(|(p, _, _)| p == "peaceful quiet retreat")
            .expect("mined the same-sentence spelling");
        assert_eq!(*count, 1);
        assert!(
            (*coh - 1.0).abs() < 1e-6,
            "same-sentence coherence should be 1.0, got {coh}"
        );
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
    fn identifier_fragments_are_not_candidates() {
        let e = Engine::in_memory().unwrap();
        // Every uppercase run here is part of a larger identifier. Splitting on
        // non-alphanumerics turns each into an acronym-shaped token, which is
        // how streaming command output produced hundreds of junk candidates.
        let out = e
            .analyze("export CLAUDE_PLUGIN_ROOT=$HOME/x and read plugins/rq/SKILL.md")
            .unwrap();
        assert!(out.candidates.is_empty(), "got {:?}", out.candidates);
        assert!(e.candidate_counts().unwrap().is_empty());
    }

    #[test]
    fn a_real_acronym_beside_an_identifier_still_counts() {
        let e = Engine::in_memory().unwrap();
        // The filter is per word, so one path doesn't mute the whole line.
        // ZQX is unseeded, so it can only show up as a candidate.
        let out = e
            .analyze("the ZQX is tracked in src/DIR_NAME/x.rs")
            .unwrap();
        assert_eq!(out.candidates, vec!["ZQX".to_string()]);
    }

    #[test]
    fn an_underscore_binds_a_name_together_rather_than_splitting_it() {
        let e = Engine::in_memory().unwrap();
        // Whole identifier is 18 characters, so it can't be acronym-shaped —
        // no need for a special case, it just stops being three tokens.
        let out = e.analyze("we set CLAUDE_PLUGIN_ROOT here").unwrap();
        assert!(out.candidates.is_empty(), "got {:?}", out.candidates);
    }

    #[test]
    fn an_underscore_wrapping_an_acronym_is_not_part_of_it() {
        let e = Engine::in_memory().unwrap();
        // Markdown emphasis. Only an underscore *between* two parts glues them,
        // so _ZQX_ is still the acronym it wraps.
        let out = e.analyze("the _ZQX_ number looks wrong").unwrap();
        assert_eq!(out.candidates, vec!["ZQX".to_string()]);
    }

    #[test]
    fn a_sentence_final_acronym_is_not_mistaken_for_a_filename() {
        let e = Engine::in_memory().unwrap();
        // "MVP." ends a sentence; only a dot *binding two parts* marks an
        // identifier, so the trailing period must not disqualify it.
        let out = e.analyze("we finally shipped the MVP.").unwrap();
        assert_eq!(out.candidates, vec!["MVP".to_string()]);
    }

    #[test]
    fn all_caps_prose_surfaces_no_candidates() {
        let e = Engine::in_memory().unwrap();
        // Every word is 2–6 uppercase letters; without the all-caps guard each
        // would be flagged as an acronym candidate.
        let out = e.analyze("SHIP THE NEW MVP TODAY").unwrap();
        assert!(out.candidates.is_empty());
        // And the flood isn't quietly recorded either.
        assert!(e.candidate_counts().unwrap().is_empty());
    }

    #[test]
    fn a_lone_all_caps_token_is_still_a_candidate() {
        let e = Engine::in_memory().unwrap();
        // One token has no surrounding lowercase to contrast against, so the
        // guard doesn't apply — a bare acronym lookup still works.
        let out = e.analyze("MVP").unwrap();
        assert!(out.candidates.contains(&"MVP".to_string()));
    }

    #[test]
    fn an_ignored_acronym_is_neither_expanded_nor_flagged() {
        let e = Engine::in_memory().unwrap();
        e.ignore_acronym("OKR").unwrap(); // a seeded, known acronym
        let out = e.analyze("review the OKR and the MVP today").unwrap();
        assert!(!out.expansions.iter().any(|r| r.acronym == "OKR")); // muted, dormant
        assert!(!out.candidates.contains(&"OKR".to_string()));
        assert!(out.candidates.contains(&"MVP".to_string())); // others unaffected
    }

    #[test]
    fn ignoring_an_acronym_suppresses_candidate_tracking_and_mining() {
        let e = Engine::in_memory().unwrap();
        e.ignore_acronym("MVP").unwrap();
        let out = e.analyze("the MVP means a minimum viable product").unwrap();
        assert!(!out.candidates.contains(&"MVP".to_string())); // not surfaced
        assert!(e.potentials_for("MVP").unwrap().is_empty()); // not mined
        assert!(
            e.candidate_counts()
                .unwrap()
                .iter()
                .all(|(a, _)| a != "MVP")
        ); // not tracked
    }

    #[test]
    fn unignoring_reactivates_an_acronym() {
        let e = Engine::in_memory().unwrap();
        e.ignore_acronym("OKR").unwrap();
        assert!(
            !e.analyze("the OKR review")
                .unwrap()
                .expansions
                .iter()
                .any(|r| r.acronym == "OKR")
        );
        assert!(e.unignore_acronym("OKR").unwrap());
        assert!(
            e.analyze("the OKR review")
                .unwrap()
                .expansions
                .iter()
                .any(|r| r.acronym == "OKR")
        );
    }

    #[test]
    fn a_warm_engine_expands_an_acronym_added_out_of_band() {
        let db = std::env::temp_dir().join(format!("ae-oob-{}.db", std::process::id()));
        let rm = || {
            for ext in ["", "-wal", "-shm"] {
                let _ = std::fs::remove_file(format!("{}{ext}", db.display()));
            }
        };
        rm();
        let warm = Engine::new(Store::open(&db).unwrap(), Box::new(HashEmbedder::new())).unwrap();
        warm.analyze("nothing yet").unwrap();
        // `ae add` from another process: its own connection, same file.
        Store::open(&db)
            .unwrap()
            .add_entry("ZQX", "Zebra Quilt Exchange", "user")
            .unwrap();
        let out = warm.analyze("ship the ZQX today").unwrap();
        rm();
        assert!(out.expansions.iter().any(|r| r.acronym == "ZQX"), "{out:?}");
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

    // --- candidate noise: shouted words, numbered labels, stoplist ---

    fn word_set(list: &[&str]) -> std::collections::HashSet<String> {
        list.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn a_shouted_ordinary_word_is_noise_when_a_wordlist_exists() {
        let words = word_set(&["not", "where", "from", "post", "warn"]);
        for token in ["NOT", "WHERE", "FROM", "POST", "WARN"] {
            assert!(is_noise(token, Some(&words)), "{token} should be noise");
        }
        // Not a word → still a candidate.
        assert!(!is_noise("ZQX", Some(&words)));
    }

    #[test]
    fn without_a_wordlist_words_stay_candidates() {
        // No system wordlist (e.g. a bare Linux box): degrade to no word filter
        // rather than guessing.
        assert!(!is_noise("WHERE", None));
        assert!(!is_noise("ZQX", None));
    }

    #[test]
    fn numbered_labels_are_noise_but_digit_acronyms_are_not() {
        for token in ["V1", "V12", "L4", "Q1", "H1", "P0", "FY27", "FY2027"] {
            assert!(is_noise(token, None), "{token} should be noise");
        }
        for token in ["S3", "EC2", "K8S", "P2P", "B2B", "MP3", "E2E"] {
            assert!(!is_noise(token, None), "{token} is a real acronym");
        }
    }

    #[test]
    fn stoplisted_tokens_are_noise_without_a_wordlist() {
        for token in ["INFO", "DEBUG", "TODO", "FIXME"] {
            assert!(is_noise(token, None), "{token} should be noise");
        }
    }

    #[test]
    fn noise_tokens_are_neither_surfaced_nor_recorded() {
        let e = Engine::in_memory().unwrap();
        // Only the stoplist and label rules — no wordlist dependence here.
        let out = e
            .analyze("bumped to V1 in Q1 FY27 with INFO logs for the ZQX bucket")
            .unwrap();
        assert_eq!(out.candidates, vec!["ZQX".to_string()]);
        let recorded: Vec<String> = e
            .candidate_counts()
            .unwrap()
            .into_iter()
            .map(|(a, _)| a)
            .collect();
        assert_eq!(recorded, vec!["ZQX".to_string()]);
    }

    #[test]
    fn a_declared_noise_token_is_still_mined() {
        let e = Engine::in_memory().unwrap();
        // The filter only hides candidates; a declared acronym is mined from
        // the watch list regardless.
        e.declare_acronym("TODO").unwrap();
        let out = e.analyze("we track our daily output here").unwrap();
        assert!(out.candidates.is_empty());
        assert!(!e.potentials_for("TODO").unwrap().is_empty());
    }

    // --- Match confidence: P(expansion | acronym, context) × validity ---

    /// A transparent embedder for scoring tests: "physical"/"therapy" and
    /// "part"/"time" each own a dimension, every other word shares a third.
    struct KeywordEmbedder;

    impl Embedder for KeywordEmbedder {
        fn embed(&self, text: &str) -> Vec<f32> {
            let mut v = vec![0.0; crate::embed::EMBED_DIMS];
            for word in text
                .split(|c: char| !c.is_alphanumeric())
                .filter(|w| !w.is_empty())
            {
                let dim = match word.to_lowercase().as_str() {
                    "physical" | "therapy" => 0,
                    "part" | "time" => 1,
                    _ => 2,
                };
                v[dim] += 1.0;
            }
            v
        }

        fn kind(&self) -> &'static str {
            "keyword"
        }
    }

    /// An engine over an empty store holding just `entries`, all verified.
    fn engine_with(entries: &[(&str, &str)]) -> Engine {
        let store = Store::open_in_memory().unwrap();
        for (acronym, expansion) in entries {
            store.add_entry(acronym, expansion, "user").unwrap();
        }
        Engine::new(store, Box::new(KeywordEmbedder)).unwrap()
    }

    fn matches_for(e: &Engine, text: &str, acronym: &str) -> Vec<MatchCandidate> {
        e.expand_only(text)
            .unwrap()
            .expansions
            .into_iter()
            .find(|r| r.acronym == acronym)
            .map(|r| r.matches)
            .unwrap_or_default()
    }

    /// The exposed (finding) confidence of each expansion of `acronym`.
    fn exposed_for(e: &Engine, text: &str, acronym: &str) -> Vec<f32> {
        e.expand_only(text)
            .unwrap()
            .findings()
            .into_iter()
            .filter(|f| f.kind == "expansion" && f.acronym == acronym)
            .filter_map(|f| f.confidence)
            .collect()
    }

    #[test]
    fn a_verified_sole_expansion_is_fully_confident() {
        let e = engine_with(&[("SLO", "Service Level Objective")]);
        assert_eq!(exposed_for(&e, "is the SLO burning down", "SLO"), [1.0]);
    }

    #[test]
    fn an_inline_sole_expansion_scores_its_validity() {
        let e = Engine::in_memory().unwrap();
        e.analyze("The ZQ (Zebra Queue) is deep.").unwrap();
        assert_eq!(exposed_for(&e, "Drain the ZQ now.", "ZQ"), [0.9]);
    }

    #[test]
    fn a_sole_expansion_needs_no_inference() {
        let (e, calls) = counting_engine();
        e.expand_only("Check the OKR dashboard.").unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn sibling_confidences_form_a_distribution() {
        let e = engine_with(&[("PT", "Part Time"), ("PT", "Physical Therapy")]);
        let matches = matches_for(&e, "the PT plan", "PT");
        let total: f32 = matches.iter().map(|m| m.confidence).sum();
        assert!((total - 1.0).abs() < 1e-4, "shares sum to {total}");
        let exposed: f32 = exposed_for(&e, "the PT plan", "PT").iter().sum();
        assert!((exposed - 1.0).abs() <= 0.02, "exposed sum {exposed}");
    }

    #[test]
    fn the_sibling_that_fits_the_context_ranks_first() {
        let e = engine_with(&[("PT", "Part Time"), ("PT", "Physical Therapy")]);
        for (text, want) in [
            (
                "booked physical therapy for my knee, PT at noon",
                "Physical Therapy",
            ),
            ("the PT role is part time hours", "Part Time"),
        ] {
            let m = matches_for(&e, text, "PT");
            assert_eq!(m[0].expansion, want, "{text}");
            assert!(m[0].confidence > 0.5, "{text}: {}", m[0].confidence);
        }
    }

    #[test]
    fn a_sibling_without_contexts_is_not_penalized() {
        // No recorded context reads the same as the first one it would record:
        // an embedding of its own phrase.
        let e = engine_with(&[("PT", "Part Time"), ("PT", "Physical Therapy")]);
        let text = "physical PT plan";
        let before: Vec<f32> = matches_for(&e, text, "PT")
            .iter()
            .map(|m| m.confidence)
            .collect();
        let (id, phrase, _) = e.store.expansions_for("PT").unwrap().remove(1);
        e.store
            .add_context(id, &compress_matryoshka_vector(&e.embedder.embed(&phrase)))
            .unwrap();
        let after: Vec<f32> = matches_for(&e, text, "PT")
            .iter()
            .map(|m| m.confidence)
            .collect();
        assert_eq!(before.len(), 2);
        for (b, a) in before.iter().zip(&after) {
            assert!((b - a).abs() < 1e-6, "{before:?} vs {after:?}");
        }
    }

    #[test]
    fn an_unaligned_extraction_is_speculative_not_confirmed() {
        let e = Engine::in_memory().unwrap();
        let out = e.analyze("we met Jane Doe (ZQX) today").unwrap();
        // Still the record of what was seen, and ZQX stays undefined.
        assert!(out.extractions.iter().any(|x| x.acronym == "ZQX"));
        assert!(out.candidates.contains(&"ZQX".to_string()));
        // Not in the dictionary, so it never expands; `suggest` has it.
        assert!(matches_for(&e, "ship the ZQX", "ZQX").is_empty());
        assert!(
            e.potentials_for("ZQX")
                .unwrap()
                .iter()
                .any(|(p, _)| p == "we met jane doe")
        );
    }
}
