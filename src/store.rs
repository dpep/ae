//! Embedded SQLite storage: the acronym dictionary and the 64-d context store.
//!
//! Vectors are stored as raw little-endian `f32` BLOBs rather than in a
//! `sqlite-vec` `vec0` virtual table — the bundled SQLite doesn't carry that
//! extension, and at this corpus size cosine ranking in Rust over the candidate
//! set is simpler and fast. The API hides the representation, so swapping in
//! `vec0` later is a storage-layer change only. See docs/ROADMAP.md.

use std::path::Path;

use rusqlite::{Connection, OptionalExtension, Result, Transaction, TransactionBehavior, params};

use serde::Serialize;

use crate::mrl::MRL_DIMS;

const SCHEMA: &str = "
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- One row per (acronym, expansion). `source` places each on the validity
-- continuum: 'user' (a human asserted it — verified) > 'inline' (defined in the
-- text) > 'mined' (speculative — initials match). `count`/`coh_sum` track the
-- recurrence and context coherence behind a mined expansion (0 for confirmed
-- ones); a mined row that's later confirmed simply upgrades its source.
CREATE TABLE IF NOT EXISTS acronym_dictionary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    acronym TEXT NOT NULL,
    expansion TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'user',
    count INTEGER NOT NULL DEFAULT 0,
    coh_sum REAL NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_acronym_lookup
    ON acronym_dictionary(acronym, expansion);

CREATE TABLE IF NOT EXISTS acronym_contexts (
    acronym_id INTEGER NOT NULL REFERENCES acronym_dictionary(id),
    context_embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_context_acronym
    ON acronym_contexts(acronym_id);

-- Acronym-shaped tokens seen but not (yet) defined — the 'is this an acronym'
-- signal (per acronym, not per expansion). `source` is its provenance:
-- 'declared' (a person said it's an acronym) vs 'seen' (ae noticed it in text).
-- An acronym joins the **watch list** (we hunt its expansions) once it's
-- 'declared' or has been seen `count` >= threshold times; pruning drops
-- seldom-seen 'seen' ones and never 'declared' ones.
CREATE TABLE IF NOT EXISTS candidate_acronyms (
    acronym TEXT PRIMARY KEY,
    count INTEGER NOT NULL DEFAULT 1,
    source TEXT NOT NULL DEFAULT 'seen',
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Running-mean context embedding per candidate acronym: where it tends to be
-- used, so we can judge whether a mined phrase's context fits.
CREATE TABLE IF NOT EXISTS candidate_contexts (
    acronym TEXT PRIMARY KEY,
    mean BLOB NOT NULL,
    n INTEGER NOT NULL DEFAULT 0
);

-- Acronyms the user has muted. Kept in the DB but inert: excluded from the
-- expansion trie, the mining/watch list, suggestions, and candidate surfacing.
-- Distinct from `rm` (which deletes the dictionary rows) — `unignore` reverses
-- it and any confirmed expansions are untouched, just dormant meanwhile.
CREATE TABLE IF NOT EXISTS ignored_acronyms (
    acronym TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Small key/value store for housekeeping state, e.g. when consolidation last ran.
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
";

/// Indexes over columns the migrations in [`Store::init`] may have just added,
/// so they run after them.
fn migrated_indexes() -> String {
    format!(
        "
-- Mined rows outnumber confirmed ones by orders of magnitude; without this,
-- every confirmed-only query (run per request) scans all of them.
CREATE INDEX IF NOT EXISTS idx_acronym_source
    ON acronym_dictionary(source, acronym);

-- Just the watch list, so counting it per request doesn't scan every
-- seen-once candidate.
CREATE INDEX IF NOT EXISTS idx_candidate_watch
    ON candidate_acronyms(acronym) WHERE {WATCHED};
"
    )
}

/// The watch-list predicate. Queries must repeat it verbatim: SQLite only uses
/// a partial index whose WHERE term it can see in the query.
const WATCHED: &str = "(source = 'declared' OR count >= 3)";

/// How often an acronym must be *seen* (or be declared) before it joins the
/// watch list and we mine its expansions from unrelated text.
pub const WATCH_THRESHOLD: i64 = 3;

/// Confidence floor `prune` (and the amortized GC) discard mined expansions
/// below. `suggest` keeps a higher bar of its own.
pub const PRUNE_MIN_CONFIDENCE: f32 = 0.15;

/// Default grace before a seen-once candidate or mined expansion is eligible
/// for noise pruning — long enough for a real term to recur, short enough that
/// a streamed corpus doesn't bank months of one-offs. 7 days; override with
/// `AE_PRUNE_GRACE_SECS`.
pub const DEFAULT_PRUNE_GRACE_SECS: i64 = 7 * 24 * 60 * 60;

/// A small built-in dictionary so expansion works on a fresh database.
pub const DEFAULT_DICTIONARY: &[(&str, &str)] = &[
    ("OKR", "Objectives and Key Results"),
    ("KPI", "Key Performance Indicator"),
    ("API", "Application Programming Interface"),
    ("CLI", "Command Line Interface"),
    ("LLM", "Large Language Model"),
    ("RAM", "Random Access Memory"),
];

pub struct Store {
    conn: Connection,
}

/// What a [`Store::consolidate`] pass changed — reported by `ae prune`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ConsolidateStats {
    pub corrected: usize,
    pub merged: usize,
    pub dropped: usize,
    pub candidates: usize,
    /// Candidates [`crate::noise::is_noise`] rejects, however often seen.
    pub non_acronyms: usize,
    /// Mined expansions dropped along with those candidates.
    pub non_acronym_expansions: usize,
    /// `inline` rows dropped for not spelling their acronym ([`Store::prune`] only).
    pub unaligned: usize,
    /// Confirmed case-only duplicates merged away ([`Store::prune`] only).
    pub case_merged: usize,
}

/// Confirmed rows the current rules wouldn't have written — see [`Store::prune`].
struct StaleConfirmed {
    /// `inline` rows whose definition doesn't spell the acronym.
    unaligned: Vec<i64>,
    /// `(dropped, kept)`: a case-only duplicate and the row it merges into.
    duplicates: Vec<(i64, i64)>,
}

/// Whether `expansion`'s initials spell `acronym` — the rule an inline
/// definition must pass to be confirmed.
fn spells(acronym: &str, expansion: &str) -> bool {
    crate::learn::is_aligned(&crate::types::Extraction {
        acronym: acronym.to_string(),
        extracted_definition: expansion.to_string(),
        pattern_type: String::new(),
        confidence: 0.0,
    })
}

/// Dictionary health for `ae stats`: counts, and shares derived from them.
/// Field names are the JSON contract.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DictionaryStats {
    /// Confirmed (`user` + `inline`) expansions, and the acronyms they cover.
    pub confirmed: usize,
    pub acronyms: usize,
    /// `user` rows. The schema can't tell a seeded default from one you added,
    /// so `builtin` counts those still matching [`DEFAULT_DICTIONARY`].
    pub verified: usize,
    pub builtin: usize,
    pub verified_share: Option<f64>,
    /// `inline` rows, and how many have initials that spell their acronym.
    pub inline: usize,
    pub aligned: usize,
    pub aligned_share: Option<f64>,
    /// Speculative rows. Not checked for alignment: mining only finds phrases
    /// whose initials spell the acronym.
    pub mined: usize,
    pub candidates: usize,
    pub watched: usize,
}

/// `n / d` to two decimals — the precision a share of small counts has — or
/// `None` when there is nothing to divide.
fn share(n: usize, d: usize) -> Option<f64> {
    (d > 0).then(|| (n as f64 / d as f64 * 100.0).round() / 100.0)
}

impl Store {
    /// Open (creating if needed) a database at `path` and apply the schema.
    /// Parent directories are created if missing.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    /// An ephemeral in-memory database — used by the in-process fallback when
    /// no persistent DB is configured, and by tests.
    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch(SCHEMA)?;
        // Migrations for older DBs (the "duplicate column" error on fresh DBs
        // that already have the column is harmless and ignored).
        for column in [
            "source TEXT NOT NULL DEFAULT 'user'",
            "count INTEGER NOT NULL DEFAULT 0",
            "coh_sum REAL NOT NULL DEFAULT 0",
            "last_seen TIMESTAMP",
        ] {
            let _ = conn.execute(
                &format!("ALTER TABLE acronym_dictionary ADD COLUMN {column}"),
                [],
            );
        }
        let _ = conn.execute(
            "ALTER TABLE candidate_acronyms ADD COLUMN source TEXT NOT NULL DEFAULT 'seen'",
            [],
        );
        conn.execute_batch(&migrated_indexes())?;
        Ok(Self { conn })
    }

    /// Insert an `(acronym, expansion)` pair from `source` (`"user"`, `"inline"`,
    /// or `"mined"`), returning its row id. Expansions differing only in case
    /// are the same row. Idempotent; a re-add only *upgrades* the source if the
    /// new one is stronger (mined < inline < user), and then its casing wins
    /// too — so `ae add` can correct a learned row's casing. Defining a
    /// confirmed expansion also un-mutes the acronym — an explicit definition
    /// reactivates one that had been ignored.
    pub fn add_entry(&self, acronym: &str, expansion: &str, source: &str) -> Result<i64> {
        let acronym = acronym.trim().to_uppercase();
        let expansion = expansion.trim();
        // NOCASE folds ASCII only. The strongest sibling first, so a mined
        // lowercase copy never wins over a confirmed row; then exact casing,
        // for legacy confirmed siblings.
        let existing: Option<(i64, String)> = self
            .conn
            .query_row(
                "SELECT id, source FROM acronym_dictionary
                 WHERE acronym = ?1 AND expansion = ?2 COLLATE NOCASE
                 ORDER BY CASE source WHEN 'user' THEN 2 WHEN 'inline' THEN 1 ELSE 0 END DESC,
                          expansion = ?2 DESC, id
                 LIMIT 1",
                params![acronym, expansion],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        let id = match existing {
            Some((id, current)) => {
                if source_rank(source) > source_rank(&current) {
                    // A sibling already spelled the new way ranks below this
                    // one (else it was picked) — speculation, which carries no
                    // contexts. Absorb it rather than collide on the index.
                    self.conn.execute(
                        "DELETE FROM acronym_dictionary
                         WHERE acronym = ?1 AND expansion = ?2 AND id != ?3",
                        params![acronym, expansion, id],
                    )?;
                    self.conn.execute(
                        "UPDATE acronym_dictionary SET source = ?2, expansion = ?3 WHERE id = ?1",
                        params![id, source, expansion],
                    )?;
                }
                id
            }
            None => {
                self.conn.execute(
                    "INSERT INTO acronym_dictionary (acronym, expansion, source) VALUES (?1, ?2, ?3)",
                    params![acronym, expansion, source],
                )?;
                self.conn.last_insert_rowid()
            }
        };
        // It's confirmed now, so it's no longer an open candidate / speculation,
        // nor muted — an explicit definition reactivates an ignored acronym.
        self.clear_candidate(&acronym)?;
        self.unignore_acronym(&acronym)?;
        Ok(id)
    }

    /// Record one sighting of an undefined acronym-shaped token, bumping its
    /// occurrence count.
    pub fn record_candidate(&self, acronym: &str) -> Result<()> {
        let acronym = acronym.trim().to_uppercase();
        if acronym.is_empty() || self.is_ignored(&acronym)? {
            return Ok(());
        }
        self.conn.execute(
            "INSERT INTO candidate_acronyms (acronym, count) VALUES (?1, 1)
             ON CONFLICT(acronym) DO UPDATE SET count = count + 1, last_seen = CURRENT_TIMESTAMP",
            params![acronym],
        )?;
        Ok(())
    }

    /// Declare `acronym` to be an acronym (`source = 'declared'`) — it joins the
    /// watch list immediately and is never pruned, even before it's seen often.
    pub fn declare_acronym(&self, acronym: &str) -> Result<()> {
        let acronym = acronym.trim().to_uppercase();
        if acronym.is_empty() {
            return Ok(());
        }
        self.conn.execute(
            "INSERT INTO candidate_acronyms (acronym, source) VALUES (?1, 'declared')
             ON CONFLICT(acronym) DO UPDATE SET source = 'declared'",
            params![acronym],
        )?;
        Ok(())
    }

    /// The watch list: acronyms we hunt expansions for — declared, or seen at
    /// least [`WATCH_THRESHOLD`] times (promoted from noise to "of interest").
    pub fn watch_list(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT acronym FROM candidate_acronyms
             WHERE {WATCHED} AND acronym NOT IN (SELECT acronym FROM ignored_acronyms)"
        ))?;
        let rows = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// How many acronyms are on the watch list — a cheap signal for whether the
    /// mineable set changed (vs rebuilding to find out).
    pub fn watch_list_count(&self) -> Result<i64> {
        self.conn.query_row(
            &format!(
                "SELECT COUNT(*) FROM candidate_acronyms
                 WHERE {WATCHED} AND acronym NOT IN (SELECT acronym FROM ignored_acronyms)"
            ),
            [],
            |row| row.get(0),
        )
    }

    /// Candidate acronyms with their `(count, source)`, most-seen first — for
    /// the `candidates` view (provenance + watch state).
    pub fn candidates_detailed(&self) -> Result<Vec<(String, i64, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT acronym, count, source FROM candidate_acronyms ORDER BY count DESC, acronym",
        )?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Candidate acronyms with their occurrence counts, most-seen first.
    pub fn candidates(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT acronym, count FROM candidate_acronyms ORDER BY count DESC, acronym",
        )?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    fn clear_candidate(&self, acronym: &str) -> Result<()> {
        let acronym = acronym.trim().to_uppercase();
        self.conn.execute(
            "DELETE FROM candidate_acronyms WHERE acronym = ?1",
            params![acronym],
        )?;
        // Drop the *speculative* rows only; confirmed expansions stay.
        self.conn.execute(
            "DELETE FROM acronym_dictionary WHERE acronym = ?1 AND source = 'mined'",
            params![acronym],
        )?;
        self.conn.execute(
            "DELETE FROM candidate_contexts WHERE acronym = ?1",
            params![acronym],
        )?;
        Ok(())
    }

    /// Mute `acronym`: keep any confirmed expansions but make it inert —
    /// excluded from the trie, mining, suggestions, and candidate surfacing. The
    /// speculative trail (open candidate, mined rows, candidate context) is
    /// cleared so it goes quiet at once. Idempotent; case-insensitive.
    pub fn ignore_acronym(&self, acronym: &str) -> Result<()> {
        let acronym = acronym.trim().to_uppercase();
        if acronym.is_empty() {
            return Ok(());
        }
        self.conn.execute(
            "INSERT OR IGNORE INTO ignored_acronyms (acronym) VALUES (?1)",
            params![acronym],
        )?;
        // Drop only the speculative state; confirmed expansions stay (dormant).
        self.clear_candidate(&acronym)?;
        Ok(())
    }

    /// Un-mute `acronym`, reactivating it. Returns whether it had been ignored.
    pub fn unignore_acronym(&self, acronym: &str) -> Result<bool> {
        let acronym = acronym.trim().to_uppercase();
        let n = self.conn.execute(
            "DELETE FROM ignored_acronyms WHERE acronym = ?1",
            params![acronym],
        )?;
        Ok(n > 0)
    }

    /// The set of muted acronyms (uppercase) — consulted to keep them inert.
    pub fn ignored_set(&self) -> Result<std::collections::HashSet<String>> {
        let mut stmt = self.conn.prepare("SELECT acronym FROM ignored_acronyms")?;
        let rows = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<_>>()?;
        Ok(rows)
    }

    /// Muted acronyms, alphabetically — for `ae ignore` with no argument.
    pub fn ignored_acronyms(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT acronym FROM ignored_acronyms ORDER BY acronym")?;
        let rows = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<_>>()?;
        Ok(rows)
    }

    /// How many acronyms are muted — folded into the mining-trie signature so
    /// ignoring one invalidates the cache (and a warm daemon rebuilds).
    pub fn ignored_count(&self) -> Result<i64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM ignored_acronyms", [], |row| {
                row.get(0)
            })
    }

    /// Whether `acronym` is muted (case-insensitive).
    fn is_ignored(&self, acronym: &str) -> Result<bool> {
        self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM ignored_acronyms WHERE acronym = ?1)",
            params![acronym.trim().to_uppercase()],
            |row| row.get(0),
        )
    }

    /// Record one sighting of a speculative (`mined`) expansion, with the vector
    /// coherence of the context it was mined from (accumulated into `coh_sum`).
    /// A pair that's already confirmed keeps its stronger source.
    pub fn record_potential(&self, acronym: &str, expansion: &str, coherence: f32) -> Result<()> {
        let acronym = acronym.trim().to_uppercase();
        let expansion = expansion.trim().to_lowercase();
        if acronym.is_empty() || expansion.is_empty() || self.is_ignored(&acronym)? {
            return Ok(());
        }
        // last_seen is explicit: a column added by migration has no default.
        self.conn.execute(
            "INSERT INTO acronym_dictionary (acronym, expansion, source, count, coh_sum, last_seen)
             VALUES (?1, ?2, 'mined', 1, ?3, CURRENT_TIMESTAMP)
             ON CONFLICT(acronym, expansion) DO UPDATE
               SET count = count + 1, coh_sum = coh_sum + ?3, last_seen = CURRENT_TIMESTAMP",
            params![acronym, expansion, coherence as f64],
        )?;
        Ok(())
    }

    /// Rename a mined expansion, merging into `to` if that row already exists.
    /// Keeps the older evidence's `last_seen`, so a corrected row stays as
    /// prunable as it was rather than looking freshly seen.
    fn move_potential(&self, acronym: &str, from: &str, to: &str) -> Result<()> {
        // `WHERE` disambiguates INSERT…SELECT from the upsert's ON clause.
        self.conn.execute(
            "INSERT INTO acronym_dictionary (acronym, expansion, source, count, coh_sum, last_seen)
             SELECT acronym, ?3, 'mined', count, coh_sum, last_seen FROM acronym_dictionary
             WHERE acronym = ?1 AND expansion = ?2 AND source = 'mined'
             ON CONFLICT(acronym, expansion) DO UPDATE
               SET count = count + excluded.count,
                   coh_sum = coh_sum + excluded.coh_sum,
                   last_seen = MAX(last_seen, excluded.last_seen)",
            params![acronym, from, to],
        )?;
        self.delete_potential(acronym, from)?;
        Ok(())
    }

    /// Spell-correct every mined expansion against `words` (see
    /// [`crate::spell::correct`]), returning how many were rewritten.
    fn spell_correct_potentials(&self, words: &std::collections::HashSet<String>) -> Result<usize> {
        let mut corrected = 0;
        for (acronym, expansion, _, _) in self.all_potentials()? {
            let fixed = crate::spell::correct(&expansion, words);
            if fixed != expansion {
                self.move_potential(&acronym, &expansion, &fixed)?;
                corrected += 1;
            }
        }
        Ok(corrected)
    }

    /// Speculative `(expansion, count, coh_sum)` rows for one acronym.
    pub fn potentials_for(&self, acronym: &str) -> Result<Vec<(String, i64, f64)>> {
        let acronym = acronym.trim().to_uppercase();
        let mut stmt = self.conn.prepare(
            "SELECT expansion, count, coh_sum FROM acronym_dictionary
             WHERE acronym = ?1 AND source = 'mined' ORDER BY count DESC, expansion",
        )?;
        let rows = stmt
            .query_map(params![acronym], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// All speculative `(acronym, expansion, count, coh_sum)` rows, for `suggest`.
    pub fn all_potentials(&self) -> Result<Vec<(String, String, i64, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT acronym, expansion, count, coh_sum FROM acronym_dictionary
             WHERE source = 'mined' ORDER BY acronym, count DESC, expansion",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Distinct acronyms that have any speculative expansions.
    pub fn distinct_potential_acronyms(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT acronym FROM acronym_dictionary WHERE source = 'mined'")?;
        let rows = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Delete one speculative expansion. Returns the number of rows removed.
    pub fn delete_potential(&self, acronym: &str, expansion: &str) -> Result<usize> {
        let n = self.conn.execute(
            "DELETE FROM acronym_dictionary
             WHERE acronym = ?1 AND expansion = ?2 AND source = 'mined'",
            params![
                acronym.trim().to_uppercase(),
                expansion.trim().to_lowercase()
            ],
        )?;
        Ok(n)
    }

    /// Merge prefix-compatible expansions of `acronym` into the most complete
    /// form (e.g. "min viable product" → "minimum viable product"), summing
    /// counts and coherence. Returns how many rows were merged away. The merged
    /// row keeps the *newest* `last_seen` of its cluster, so dedup doesn't make
    /// stale rows look freshly seen (which would dodge age-based pruning).
    pub fn dedup_potentials(&self, acronym: &str) -> Result<usize> {
        let acronym = acronym.trim().to_uppercase();
        let mut rows: Vec<(String, i64, f64, Option<String>)> = {
            let mut stmt = self.conn.prepare(
                "SELECT expansion, count, coh_sum, last_seen FROM acronym_dictionary
                 WHERE acronym = ?1 AND source = 'mined'",
            )?;
            stmt.query_map(params![acronym], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?
            .collect::<Result<Vec<_>>>()?
        };
        let original = rows.len();
        // Longest first, so the canonical form of each cluster is the fullest.
        rows.sort_by_key(|r| std::cmp::Reverse(r.0.chars().count()));

        let mut clusters: Vec<(String, i64, f64, Option<String>)> = Vec::new();
        for (expansion, count, coh, last_seen) in rows {
            match clusters.iter_mut().find(|c| similar(&c.0, &expansion)) {
                Some(c) => {
                    c.1 += count;
                    c.2 += coh;
                    // ISO-8601 text sorts chronologically.
                    if last_seen > c.3 {
                        c.3 = last_seen;
                    }
                }
                None => clusters.push((expansion, count, coh, last_seen)),
            }
        }
        if clusters.len() == original {
            return Ok(0);
        }
        self.conn.execute(
            "DELETE FROM acronym_dictionary WHERE acronym = ?1 AND source = 'mined'",
            params![acronym],
        )?;
        for (expansion, count, coh, last_seen) in &clusters {
            self.conn.execute(
                "INSERT INTO acronym_dictionary (acronym, expansion, source, count, coh_sum, last_seen)
                 VALUES (?1, ?2, 'mined', ?3, ?4, ?5)",
                params![acronym, expansion, count, coh, last_seen],
            )?;
        }
        Ok(original - clusters.len())
    }

    /// The `(acronym, expansion)` mined rows seen within the last `grace_secs` —
    /// spared from the low-confidence drop so a freshly mined expansion gets time
    /// to recur before it's judged.
    pub fn recent_potentials(
        &self,
        grace_secs: i64,
    ) -> Result<std::collections::HashSet<(String, String)>> {
        let cutoff = format!("-{} seconds", grace_secs.max(0));
        let mut stmt = self.conn.prepare(
            "SELECT acronym, expansion FROM acronym_dictionary
             WHERE source = 'mined' AND last_seen > datetime('now', ?1)",
        )?;
        let rows = stmt
            .query_map(params![cutoff], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<std::collections::HashSet<_>>>()?;
        Ok(rows)
    }

    /// Drop noise candidates — seen once, with nothing mined, not declared, and
    /// not seen within the last `grace_secs` (so a just-seen token isn't yanked
    /// out from under the user) — plus any orphaned context rows. Returns the
    /// number of candidates removed.
    pub fn prune_noise_candidates(&self, grace_secs: i64) -> Result<usize> {
        let cutoff = format!("-{} seconds", grace_secs.max(0));
        let n = self.conn.execute(
            "DELETE FROM candidate_acronyms
             WHERE count <= 1 AND source != 'declared'
               AND last_seen <= datetime('now', ?1)
               AND acronym NOT IN
                   (SELECT acronym FROM acronym_dictionary WHERE source = 'mined')",
            params![cutoff],
        )?;
        self.conn.execute(
            "DELETE FROM candidate_contexts
             WHERE acronym NOT IN (SELECT acronym FROM candidate_acronyms)",
            [],
        )?;
        Ok(n)
    }

    /// Drop undeclared candidates that aren't acronyms by [`crate::noise::is_noise`]
    /// (recorded before that rule existed), with their mined expansions and
    /// context. No grace period: the verdict is about the token, not its age.
    /// Returns `(candidates, mined rows)` removed.
    pub fn prune_non_acronyms(
        &self,
        words: Option<&std::collections::HashSet<String>>,
    ) -> Result<(usize, usize)> {
        let mut stmt = self
            .conn
            .prepare("SELECT acronym FROM candidate_acronyms WHERE source != 'declared'")?;
        let noise: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<Vec<String>>>()?
            .into_iter()
            .filter(|a| crate::noise::is_noise(a, words))
            .collect();
        let mut mined = 0;
        for acronym in &noise {
            mined += self.conn.execute(
                "DELETE FROM acronym_dictionary WHERE acronym = ?1 AND source = 'mined'",
                params![acronym],
            )?;
            self.clear_candidate(acronym)?;
        }
        Ok((noise.len(), mined))
    }

    /// Consolidate speculation — the shared body of `ae prune` and the periodic
    /// auto-job. Candidates that aren't acronyms go first, so no effort is spent
    /// correcting their mined rows. The *quality* steps (spell-correct mined words, then merge
    /// near-duplicate expansions) run first and boost confidence by pooling
    /// evidence; the *cleanup* steps (drop low-confidence rows and seen-once
    /// noise candidates, both sparing anything within `grace_secs`) follow.
    ///
    /// One transaction, so a failure can't strand dedup's delete-and-reinsert
    /// halfway, and tens of thousands of row writes don't each commit.
    pub fn consolidate(&self, min_confidence: f32, grace_secs: i64) -> Result<ConsolidateStats> {
        let words = crate::spell::load_wordlist();
        self.in_transaction(|| self.consolidate_rows(words.as_ref(), min_confidence, grace_secs))
    }

    /// `ae prune`: bring confirmed rows up to the current rules, then
    /// [`Self::consolidate`]. Drops `inline` definitions that don't spell their
    /// acronym (learned before only aligned ones were confirmed), and merges
    /// case-only duplicates by [`Self::add_entry`]'s rule — strongest source,
    /// then the oldest row's casing — moving their contexts to the survivor.
    ///
    /// The periodic job runs only `consolidate`: deleting dictionary entries is
    /// left to an explicit prune, which backs up first
    /// ([`Self::stale_confirmed_count`]).
    pub fn prune(&self, min_confidence: f32, grace_secs: i64) -> Result<ConsolidateStats> {
        let words = crate::spell::load_wordlist();
        self.in_transaction(|| {
            let stale = self.stale_confirmed()?;
            for id in &stale.unaligned {
                self.conn.execute(
                    "DELETE FROM acronym_contexts WHERE acronym_id = ?1",
                    params![id],
                )?;
                self.conn
                    .execute("DELETE FROM acronym_dictionary WHERE id = ?1", params![id])?;
            }
            for (dropped, kept) in &stale.duplicates {
                self.conn.execute(
                    "UPDATE acronym_contexts SET acronym_id = ?2 WHERE acronym_id = ?1",
                    params![dropped, kept],
                )?;
                self.conn.execute(
                    "DELETE FROM acronym_dictionary WHERE id = ?1",
                    params![dropped],
                )?;
            }
            Ok(ConsolidateStats {
                unaligned: stale.unaligned.len(),
                case_merged: stale.duplicates.len(),
                ..self.consolidate_rows(words.as_ref(), min_confidence, grace_secs)?
            })
        })
    }

    /// How many confirmed rows [`Self::prune`] would drop or merge.
    pub fn stale_confirmed_count(&self) -> Result<usize> {
        let stale = self.stale_confirmed()?;
        Ok(stale.unaligned.len() + stale.duplicates.len())
    }

    fn stale_confirmed(&self) -> Result<StaleConfirmed> {
        let mut stmt = self.conn.prepare(
            "SELECT id, acronym, expansion, source FROM acronym_dictionary
             WHERE source IN ('user', 'inline')",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })?
            .collect::<Result<Vec<_>>>()?;

        let mut unaligned = Vec::new();
        let mut siblings: std::collections::HashMap<(String, String), Vec<(u8, i64)>> =
            std::collections::HashMap::new();
        for (id, acronym, expansion, source) in rows {
            if source == "inline" && !spells(&acronym, &expansion) {
                unaligned.push(id);
            } else {
                let key = (acronym, expansion.to_lowercase());
                siblings
                    .entry(key)
                    .or_default()
                    .push((source_rank(&source), id));
            }
        }
        let mut duplicates = Vec::new();
        for group in siblings.into_values().filter(|g| g.len() > 1) {
            let (_, kept) = *group
                .iter()
                .max_by_key(|(rank, id)| (*rank, std::cmp::Reverse(*id)))
                .expect("group has rows");
            duplicates.extend(
                group
                    .iter()
                    .filter(|(_, id)| *id != kept)
                    .map(|(_, id)| (*id, kept)),
            );
        }
        Ok(StaleConfirmed {
            unaligned,
            duplicates,
        })
    }

    fn consolidate_rows(
        &self,
        words: Option<&std::collections::HashSet<String>>,
        min_confidence: f32,
        grace_secs: i64,
    ) -> Result<ConsolidateStats> {
        // 0. Drop candidates that aren't acronyms, and what was mined for them.
        let (non_acronyms, non_acronym_expansions) =
            self.prune_non_acronyms(crate::spell::system_short_words())?;
        // 1. Spell-correct mined words against the system list (if installed), so
        //    fixed forms then merge in dedup.
        let corrected = match words {
            Some(words) => self.spell_correct_potentials(words)?,
            None => 0,
        };
        // 2. Merge near-duplicate expansions.
        let mut merged = 0;
        for acronym in self.distinct_potential_acronyms()? {
            merged += self.dedup_potentials(&acronym)?;
        }
        // 3. Drop mined rows that never recurred or score below the floor,
        //    sparing recently seen ones. Recurrence has to be checked on its
        //    own: coherence defaults to 1.0 without history, which lifts a
        //    seen-once row above any sane floor.
        let recent = self.recent_potentials(grace_secs)?;
        let all = self.all_potentials()?;
        let mut totals: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        for (acronym, _, count, _) in &all {
            *totals.entry(acronym.clone()).or_insert(0) += count;
        }
        let mut dropped = 0;
        for (acronym, expansion, count, coh) in all {
            if (count <= 1 || confidence(count, coh, totals[&acronym]) < min_confidence)
                && !recent.contains(&(acronym.clone(), expansion.clone()))
            {
                dropped += self.delete_potential(&acronym, &expansion)?;
            }
        }
        // 4. Clear noise candidates.
        let candidates = self.prune_noise_candidates(grace_secs)?;

        self.mark_consolidated()?;
        Ok(ConsolidateStats {
            corrected,
            merged,
            dropped,
            candidates,
            non_acronyms,
            non_acronym_expansions,
            ..ConsolidateStats::default()
        })
    }

    /// Run `f` in one write transaction, rolled back if it fails. Immediate, so
    /// a concurrent writer waits on the busy timeout up front instead of
    /// failing a read-to-write lock upgrade midway.
    fn in_transaction<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let tx = Transaction::new_unchecked(&self.conn, TransactionBehavior::Immediate)?;
        let out = f()?;
        tx.commit()?;
        Ok(out)
    }

    /// True if consolidation hasn't run within the last `interval_secs` (or has
    /// never run) — the cadence gate for the auto-job.
    pub fn consolidate_due(&self, interval_secs: i64) -> Result<bool> {
        let cutoff = format!("-{} seconds", interval_secs.max(0));
        let due: i64 = self.conn.query_row(
            "SELECT COALESCE(MAX(value <= datetime('now', ?1)), 1)
             FROM meta WHERE key = 'last_consolidated'",
            params![cutoff],
            |row| row.get(0),
        )?;
        Ok(due != 0)
    }

    /// Stamp the last-consolidated time as now.
    pub fn mark_consolidated(&self) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('last_consolidated', datetime('now'))
             ON CONFLICT(key) DO UPDATE SET value = datetime('now')",
            [],
        )?;
        Ok(())
    }

    /// The running-mean context embedding for a candidate acronym, if any.
    pub fn candidate_context_mean(&self, acronym: &str) -> Result<Option<Vec<f32>>> {
        let acronym = acronym.trim().to_uppercase();
        let blob: Option<Vec<u8>> = self
            .conn
            .query_row(
                "SELECT mean FROM candidate_contexts WHERE acronym = ?1",
                params![acronym],
                |row| row.get(0),
            )
            .optional()?;
        Ok(blob.map(|b| decode(&b)))
    }

    /// Fold one context vector into a candidate acronym's running mean.
    pub fn update_candidate_context(&self, acronym: &str, vector: &[f32]) -> Result<()> {
        let acronym = acronym.trim().to_uppercase();
        let current: Option<(Vec<u8>, i64)> = self
            .conn
            .query_row(
                "SELECT mean, n FROM candidate_contexts WHERE acronym = ?1",
                params![acronym],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;

        let (mean, n) = match current {
            Some((blob, n)) => {
                let mut mean = decode(&blob);
                let n = n as f32;
                // Incremental mean: mean += (x - mean) / (n + 1).
                for (m, &x) in mean.iter_mut().zip(vector) {
                    *m += (x - *m) / (n + 1.0);
                }
                (mean, n as i64 + 1)
            }
            None => (vector.to_vec(), 1),
        };
        self.conn.execute(
            "INSERT INTO candidate_contexts (acronym, mean, n) VALUES (?1, ?2, ?3)
             ON CONFLICT(acronym) DO UPDATE SET mean = ?2, n = ?3",
            params![acronym, encode(&mean), n],
        )?;
        Ok(())
    }

    /// Seed the built-in dictionary if the table is empty. Returns the number of
    /// rows inserted. Built-ins are curated, so they count as `"user"`-verified.
    pub fn seed_defaults(&self) -> Result<usize> {
        if self.count()? > 0 {
            return Ok(0);
        }
        let mut n = 0;
        for (acr, exp) in DEFAULT_DICTIONARY {
            self.add_entry(acr, exp, "user")?;
            n += 1;
        }
        Ok(n)
    }

    /// All `(id, expansion, source)` rows for `acronym` (case-insensitive).
    pub fn expansions_for(&self, acronym: &str) -> Result<Vec<(i64, String, String)>> {
        let acronym = acronym.trim().to_uppercase();
        let mut stmt = self.conn.prepare(
            "SELECT id, expansion, source FROM acronym_dictionary
             WHERE acronym = ?1 AND source IN ('user', 'inline') ORDER BY id",
        )?;
        let rows = stmt
            .query_map(params![acronym], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Every distinct *confirmed*, non-ignored acronym — used to hydrate the
    /// expansion trie and seed the mining trie. (A mined-only acronym stays a
    /// candidate, so it isn't expanded; an ignored one is dormant.)
    pub fn all_acronyms(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT acronym FROM acronym_dictionary
             WHERE source IN ('user', 'inline')
               AND acronym NOT IN (SELECT acronym FROM ignored_acronyms)
             ORDER BY acronym",
        )?;
        let rows = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Every `(acronym, expansion, source)` row, ordered — for `list`.
    pub fn all_entries(&self) -> Result<Vec<(String, String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT acronym, expansion, source FROM acronym_dictionary
             WHERE source IN ('user', 'inline') ORDER BY acronym, expansion",
        )?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Entries whose acronym or expansion contains `query` (case-insensitive).
    pub fn search(&self, query: &str) -> Result<Vec<(String, String, String)>> {
        let pattern = format!("%{}%", query.trim());
        let mut stmt = self.conn.prepare(
            "SELECT acronym, expansion, source FROM acronym_dictionary
             WHERE source IN ('user', 'inline') AND (acronym LIKE ?1 OR expansion LIKE ?1)
             ORDER BY acronym, expansion",
        )?;
        let rows = stmt
            .query_map(params![pattern], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Delete every expansion of `acronym` (and its context vectors). Returns
    /// the number of dictionary rows removed.
    pub fn delete_acronym(&self, acronym: &str) -> Result<usize> {
        let acronym = acronym.trim().to_uppercase();
        self.conn.execute(
            "DELETE FROM acronym_contexts WHERE acronym_id IN
                 (SELECT id FROM acronym_dictionary WHERE acronym = ?1)",
            params![acronym],
        )?;
        let n = self.conn.execute(
            "DELETE FROM acronym_dictionary WHERE acronym = ?1",
            params![acronym],
        )?;
        Ok(n)
    }

    /// Delete one specific `(acronym, expansion)` pair (and its context vectors).
    pub fn delete_entry(&self, acronym: &str, expansion: &str) -> Result<usize> {
        let acronym = acronym.trim().to_uppercase();
        let expansion = expansion.trim();
        self.conn.execute(
            "DELETE FROM acronym_contexts WHERE acronym_id IN
                 (SELECT id FROM acronym_dictionary WHERE acronym = ?1 AND expansion = ?2)",
            params![acronym, expansion],
        )?;
        let n = self.conn.execute(
            "DELETE FROM acronym_dictionary WHERE acronym = ?1 AND expansion = ?2",
            params![acronym, expansion],
        )?;
        Ok(n)
    }

    /// Delete every expansion a person didn't assert — `inline` and `mined`
    /// rows, keeping `user` ones — for one acronym, or all of them with `None`.
    /// Returns the number of dictionary rows removed.
    pub fn delete_unverified(&self, acronym: Option<&str>) -> Result<usize> {
        let acronym = acronym.map(|a| a.trim().to_uppercase());
        // `?1 IS NULL` makes the acronym filter optional in one statement.
        let rows = "SELECT id FROM acronym_dictionary
                    WHERE source != 'user' AND (?1 IS NULL OR acronym = ?1)";
        self.in_transaction(|| {
            self.conn.execute(
                &format!("DELETE FROM acronym_contexts WHERE acronym_id IN ({rows})"),
                params![acronym],
            )?;
            self.conn.execute(
                &format!("DELETE FROM acronym_dictionary WHERE id IN ({rows})"),
                params![acronym],
            )
        })
    }

    /// How many rows [`Self::delete_unverified`] would remove across every
    /// acronym — so a no-op bulk remove can skip its backup.
    pub fn unverified_count(&self) -> Result<usize> {
        self.conn.query_row(
            "SELECT COUNT(*) FROM acronym_dictionary WHERE source != 'user'",
            [],
            |row| row.get(0),
        )
    }

    /// Dictionary health — see [`DictionaryStats`].
    pub fn stats(&self) -> Result<DictionaryStats> {
        let mut stmt = self.conn.prepare(
            "SELECT acronym, expansion, source FROM acronym_dictionary
             WHERE source IN ('user', 'inline')",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .collect::<Result<Vec<_>>>()?;

        let mut acronyms = std::collections::HashSet::new();
        let (mut verified, mut builtin, mut inline, mut aligned) = (0, 0, 0, 0);
        for (acronym, expansion, source) in &rows {
            acronyms.insert(acronym.as_str());
            if source == "user" {
                verified += 1;
                builtin += usize::from(DEFAULT_DICTIONARY.iter().any(|(a, e)| {
                    a.eq_ignore_ascii_case(acronym) && e.eq_ignore_ascii_case(expansion)
                }));
            } else {
                inline += 1;
                aligned += usize::from(spells(acronym, expansion));
            }
        }
        let count = |sql: &str| -> Result<usize> {
            self.conn
                .query_row(sql, [], |row| row.get::<_, i64>(0))
                .map(|n| n as usize)
        };
        Ok(DictionaryStats {
            confirmed: rows.len(),
            acronyms: acronyms.len(),
            verified,
            builtin,
            verified_share: share(verified, rows.len()),
            inline,
            aligned,
            aligned_share: share(aligned, inline),
            mined: count("SELECT COUNT(*) FROM acronym_dictionary WHERE source = 'mined'")?,
            candidates: count("SELECT COUNT(*) FROM candidate_acronyms")?,
            watched: count(&format!(
                "SELECT COUNT(*) FROM candidate_acronyms WHERE {WATCHED}"
            ))?,
        })
    }

    /// Number of *confirmed* expansions — gates default seeding.
    pub fn count(&self) -> Result<i64> {
        self.conn.query_row(
            "SELECT COUNT(*) FROM acronym_dictionary WHERE source IN ('user', 'inline')",
            [],
            |row| row.get(0),
        )
    }

    /// Write a consistent single-file snapshot of the database to `path`
    /// (SQLite `VACUUM INTO`), independent of WAL state — the backup `rm --all`
    /// takes before wiping, and `restore` reads back. `path`'s parent must exist.
    pub fn backup_to(&self, path: &std::path::Path) -> Result<()> {
        self.conn
            .execute("VACUUM INTO ?1", params![path.to_string_lossy()])?;
        Ok(())
    }

    /// Wipe every table — dictionary, contexts, candidates, speculation, and the
    /// mute list. The built-in defaults re-seed on the next open, so this is a
    /// factory reset. Returns `(dictionary_rows, candidate_rows)` cleared, for
    /// reporting. `rm --all` calls this *after* [`Self::backup_to`].
    pub fn clear_all(&self) -> Result<(i64, i64)> {
        let entries: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM acronym_dictionary", [], |row| {
                    row.get(0)
                })?;
        let candidates: i64 =
            self.conn
                .query_row("SELECT COUNT(*) FROM candidate_acronyms", [], |row| {
                    row.get(0)
                })?;
        self.conn.execute_batch(
            "DELETE FROM acronym_contexts;
             DELETE FROM acronym_dictionary;
             DELETE FROM candidate_contexts;
             DELETE FROM candidate_acronyms;
             DELETE FROM ignored_acronyms;
             DELETE FROM meta;",
        )?;
        Ok((entries, candidates))
    }

    /// Attach a compressed (64-d) context embedding to a dictionary entry.
    pub fn add_context(&self, acronym_id: i64, embedding: &[f32]) -> Result<()> {
        debug_assert_eq!(embedding.len(), MRL_DIMS);
        self.conn.execute(
            "INSERT INTO acronym_contexts (acronym_id, context_embedding) VALUES (?1, ?2)",
            params![acronym_id, encode(embedding)],
        )?;
        Ok(())
    }

    /// All context embeddings recorded for a dictionary entry.
    pub fn contexts_for(&self, acronym_id: i64) -> Result<Vec<Vec<f32>>> {
        let mut stmt = self
            .conn
            .prepare("SELECT context_embedding FROM acronym_contexts WHERE acronym_id = ?1")?;
        let rows = stmt
            .query_map(params![acronym_id], |row| {
                let blob: Vec<u8> = row.get(0)?;
                Ok(decode(&blob))
            })?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }
}

/// Strength ordering of a confirmed expansion's source.
pub fn source_rank(source: &str) -> u8 {
    match source {
        "user" => 2,
        "inline" => 1,
        _ => 0,
    }
}

/// Validity — P(this is a real expansion of the acronym) — from the source: a
/// human assertion is certain, an inline definition nearly so.
pub fn source_validity(source: &str) -> f32 {
    match source {
        "user" => 1.0,
        "inline" => 0.9,
        _ => 0.0,
    }
}

/// Blend recurrence and coherence into a `[0, 1]` confidence for a mined
/// expansion: `share` (its fraction of the acronym's sightings) and `mean_coh`
/// (average context coherence) weighted equally, then damped so a lone sighting
/// can't reach certainty. Shared by `suggest`/`prune` and the amortized GC.
pub fn confidence(count: i64, coh_sum: f64, total: i64) -> f32 {
    let count = count.max(1) as f32;
    let share = count / total.max(1) as f32;
    let mean_coh = (coh_sum as f32 / count).clamp(0.0, 1.0);
    ((0.5 * share + 0.5 * mean_coh) * (count / (count + 1.0))).clamp(0.0, 1.0)
}

/// Two mined expansions should be merged if they're prefix-compatible *or*
/// within a small edit distance (a typo/misspelling) — the fuzzy dedup `prune`
/// applies.
fn similar(a: &str, b: &str) -> bool {
    let max_len = a.chars().count().max(b.chars().count());
    prefix_compatible(a, b) || within_edits(a, b, (max_len / 8).max(1))
}

/// Whether `a` and `b` are at most `k` edits apart (Levenshtein, over chars).
/// Bails as soon as the answer is no: dedup compares every pair within an
/// acronym, and one with thousands of mined rows made the full distance a
/// multi-second stall.
fn within_edits(a: &str, b: &str, k: usize) -> bool {
    if a.chars().count().abs_diff(b.chars().count()) > k {
        return false;
    }
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j + 1] + 1)
                .min(cur[j] + 1)
                .min(prev[j] + usize::from(ca != cb));
        }
        // A row's minimum never decreases further down.
        if cur.iter().all(|&d| d > k) {
            return false;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()] <= k
}

/// Two expansions are prefix-compatible if they have the same word count and
/// each word pair is equal or one is a (≥3-char) prefix of the other — so
/// "min viable product" ≈ "minimum viable product".
fn prefix_compatible(a: &str, b: &str) -> bool {
    let (mut aw, mut bw) = (a.split_whitespace(), b.split_whitespace());
    loop {
        match (aw.next(), bw.next()) {
            (Some(x), Some(y)) if word_compatible(x, y) => {}
            (None, None) => return true,
            _ => return false,
        }
    }
}

fn word_compatible(x: &str, y: &str) -> bool {
    if x == y {
        return true;
    }
    let (short, long) = if x.len() < y.len() { (x, y) } else { (y, x) };
    short.len() >= 3 && long.starts_with(short)
}

/// Pack `f32`s into little-endian bytes.
fn encode(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes
}

/// Unpack little-endian bytes back into `f32`s.
fn decode(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks.iter().copied().map(f32::from_le_bytes).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_look_up_an_entry() {
        let s = Store::open_in_memory().unwrap();
        let id = s
            .add_entry("kpi", "Key Performance Indicator", "user")
            .unwrap();
        let rows = s.expansions_for("KPI").unwrap();
        assert_eq!(
            rows,
            vec![(
                id,
                "Key Performance Indicator".to_string(),
                "user".to_string()
            )]
        );
    }

    #[test]
    fn duplicate_pairs_are_idempotent() {
        let s = Store::open_in_memory().unwrap();
        let a = s
            .add_entry("KPI", "Key Performance Indicator", "user")
            .unwrap();
        let b = s
            .add_entry("KPI", "Key Performance Indicator", "user")
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(s.count().unwrap(), 1);
    }

    #[test]
    fn source_upgrades_but_never_downgrades() {
        let s = Store::open_in_memory().unwrap();
        s.add_entry("MVP", "Minimum Viable Product", "inline")
            .unwrap();
        s.add_entry("MVP", "Minimum Viable Product", "user")
            .unwrap(); // upgrade
        s.add_entry("MVP", "Minimum Viable Product", "inline")
            .unwrap(); // weaker, ignored
        let rows = s.expansions_for("MVP").unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].2, "user");
    }

    #[test]
    fn one_acronym_can_have_several_expansions() {
        let s = Store::open_in_memory().unwrap();
        s.add_entry("PT", "Physical Therapy", "user").unwrap();
        s.add_entry("PT", "Part Time", "user").unwrap();
        assert_eq!(s.expansions_for("PT").unwrap().len(), 2);
    }

    #[test]
    fn seeding_is_idempotent() {
        let s = Store::open_in_memory().unwrap();
        assert!(s.seed_defaults().unwrap() > 0);
        assert_eq!(s.seed_defaults().unwrap(), 0);
    }

    #[test]
    fn search_matches_acronym_or_expansion() {
        let s = Store::open_in_memory().unwrap();
        s.add_entry("KPI", "Key Performance Indicator", "user")
            .unwrap();
        s.add_entry("OKR", "Objectives and Key Results", "user")
            .unwrap();
        // Matches on the expansion text ("Key" appears in both).
        assert_eq!(s.search("key").unwrap().len(), 2);
        // Matches on the acronym.
        assert_eq!(
            s.search("kpi").unwrap(),
            vec![(
                "KPI".into(),
                "Key Performance Indicator".into(),
                "user".into()
            )]
        );
        assert!(s.search("nope").unwrap().is_empty());
    }

    #[test]
    fn delete_removes_entries_and_is_counted() {
        let s = Store::open_in_memory().unwrap();
        s.add_entry("PT", "Physical Therapy", "user").unwrap();
        s.add_entry("PT", "Part Time", "user").unwrap();
        assert_eq!(s.delete_entry("PT", "Part Time").unwrap(), 1);
        assert_eq!(s.expansions_for("PT").unwrap().len(), 1);
        assert_eq!(s.delete_acronym("PT").unwrap(), 1);
        assert!(s.expansions_for("PT").unwrap().is_empty());
        // Deleting something absent removes nothing.
        assert_eq!(s.delete_acronym("ZZ").unwrap(), 0);
    }

    #[test]
    fn candidates_count_and_clear_when_defined() {
        let s = Store::open_in_memory().unwrap();
        s.record_candidate("MVP").unwrap();
        s.record_candidate("mvp").unwrap(); // case-insensitive → same row
        s.record_candidate("ABC").unwrap();
        // MVP seen twice → ranks first.
        assert_eq!(
            s.candidates().unwrap(),
            vec![("MVP".into(), 2), ("ABC".into(), 1)]
        );
        // Defining it removes it from the candidate list.
        s.add_entry("MVP", "Minimum Viable Product", "user")
            .unwrap();
        assert_eq!(s.candidates().unwrap(), vec![("ABC".into(), 1)]);
    }

    #[test]
    fn dedup_merges_prefix_compatible_expansions() {
        let s = Store::open_in_memory().unwrap();
        s.record_potential("MVP", "min viable product", 1.0)
            .unwrap();
        s.record_potential("MVP", "minimum viable product", 1.0)
            .unwrap();
        s.record_potential("MVP", "most valuable player", 1.0)
            .unwrap();

        assert_eq!(s.dedup_potentials("MVP").unwrap(), 1); // two folded into one
        let pots = s.potentials_for("MVP").unwrap();
        assert_eq!(pots.len(), 2);
        // Canonical is the fullest form, with counts summed.
        assert!(
            pots.iter()
                .any(|(e, c, _)| e == "minimum viable product" && *c == 2)
        );
        assert!(pots.iter().any(|(e, _, _)| e == "most valuable player"));
    }

    #[test]
    fn a_legacy_database_opens_and_stamps_new_mined_rows() {
        let path = std::env::temp_dir().join(format!("ae-legacy-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&path);
        Connection::open(&path)
            .unwrap()
            .execute_batch(
                "CREATE TABLE acronym_dictionary (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     acronym TEXT NOT NULL,
                     expansion TEXT NOT NULL,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
                 CREATE TABLE candidate_acronyms (
                     acronym TEXT PRIMARY KEY,
                     count INTEGER NOT NULL DEFAULT 1);
                 INSERT INTO acronym_dictionary (acronym, expansion)
                     VALUES ('KPI', 'Key Performance Indicator');",
            )
            .unwrap();
        let opened = Store::open(&path).and_then(|s| {
            // ADD COLUMN can't carry a CURRENT_TIMESTAMP default, so a new row
            // must stamp last_seen itself or it never counts as recent.
            s.record_potential("ZQX", "zebra quilt exchange", 1.0)?;
            Ok((s.count()?, s.recent_potentials(3600)?))
        });
        let _ = std::fs::remove_file(&path);
        let (count, recent) = opened.unwrap();
        assert_eq!(count, 1);
        assert!(recent.contains(&("ZQX".into(), "zebra quilt exchange".into())));
    }

    #[test]
    fn per_request_lookups_narrow_past_mined_rows() {
        let s = Store::open_in_memory().unwrap();
        let plan = |sql: &str| -> String {
            let mut stmt = s
                .conn
                .prepare(&format!("EXPLAIN QUERY PLAN {sql}"))
                .unwrap();
            stmt.query_map([], |row| row.get::<_, String>(3))
                .unwrap()
                .collect::<Result<Vec<_>>>()
                .unwrap()
                .join("; ")
        };
        // (query, what its plan must narrow by)
        let cases = [
            (
                "SELECT COUNT(*) FROM acronym_dictionary WHERE source IN ('user', 'inline')",
                "source=?",
            ),
            // Not by source alone: that walks every mined row.
            (
                "SELECT expansion FROM acronym_dictionary WHERE acronym = 'X' AND source = 'mined'",
                "acronym=?",
            ),
            (
                &format!("SELECT COUNT(*) FROM candidate_acronyms WHERE {WATCHED}"),
                "idx_candidate_watch",
            ),
        ];
        for (sql, narrows_by) in cases {
            let p = plan(sql);
            assert!(p.contains(narrows_by), "{sql}: {p}");
        }
    }

    #[test]
    fn within_edits_matches_levenshtein_at_the_boundary() {
        // (a, b, distance)
        let cases = [
            ("kitten", "sitting", 3),
            ("", "ab", 2),
            ("abc", "abc", 0),
            ("abc", "acb", 2),
            ("flaw", "lawn", 2),
            ("café", "cafe", 1),
            ("minimum viable product", "minimum viabel product", 2),
        ];
        for (a, b, d) in cases {
            assert!(within_edits(a, b, d), "{a:?} vs {b:?} within {d}");
            assert!(within_edits(b, a, d), "{b:?} vs {a:?} within {d}");
            if d > 0 {
                assert!(
                    !within_edits(a, b, d - 1),
                    "{a:?} vs {b:?} not within {}",
                    d - 1
                );
            }
        }
    }

    #[test]
    fn confidence_rewards_recurrence_and_coherence() {
        let strong = confidence(4, 4.0, 5); // share 0.8, coherence 1.0
        let weak = confidence(1, 0.1, 5); // share 0.2, coherence 0.1
        assert!(strong > weak);
        assert!((0.0..=1.0).contains(&strong) && (0.0..=1.0).contains(&weak));
    }

    #[test]
    fn dedup_merges_a_misspelled_variant() {
        let s = Store::open_in_memory().unwrap();
        s.record_potential("MVP", "minimum viable product", 1.0)
            .unwrap();
        s.record_potential("MVP", "minimum viable prodcut", 1.0)
            .unwrap(); // typo
        assert_eq!(s.dedup_potentials("MVP").unwrap(), 1);
        assert_eq!(s.potentials_for("MVP").unwrap().len(), 1);
    }

    #[test]
    fn the_watch_list_starts_at_the_threshold() {
        let s = Store::open_in_memory().unwrap();
        for _ in 0..WATCH_THRESHOLD {
            s.record_candidate("ZQX").unwrap();
        }
        for _ in 1..WATCH_THRESHOLD {
            s.record_candidate("ZQY").unwrap();
        }
        assert_eq!(s.watch_list().unwrap(), vec!["ZQX".to_string()]);
        assert_eq!(s.watch_list_count().unwrap(), 1);
    }

    #[test]
    fn declared_acronyms_are_on_the_watch_list_and_survive_pruning() {
        let s = Store::open_in_memory().unwrap();
        s.declare_acronym("MVP").unwrap();
        s.record_candidate("XX").unwrap(); // seen once → prunable noise
        assert!(s.watch_list().unwrap().contains(&"MVP".to_string()));
        s.prune_noise_candidates(0).unwrap(); // no grace → prune immediately
        let acrs: Vec<String> = s
            .candidates()
            .unwrap()
            .into_iter()
            .map(|(a, _)| a)
            .collect();
        assert!(acrs.contains(&"MVP".to_string()) && !acrs.contains(&"XX".to_string()));
    }

    #[test]
    fn dedup_preserves_age_so_stale_rows_stay_prunable() {
        let s = Store::open_in_memory().unwrap();
        s.record_potential("MVP", "min viable product", 1.0)
            .unwrap();
        s.record_potential("MVP", "minimum viable product", 1.0)
            .unwrap();
        // Age the mined rows two hours into the past.
        s.conn
            .execute(
                "UPDATE acronym_dictionary SET last_seen = datetime('now', '-2 hours')
                 WHERE source = 'mined'",
                [],
            )
            .unwrap();
        s.dedup_potentials("MVP").unwrap();
        // The merged row kept the old timestamp → not recent within an hour.
        assert!(s.recent_potentials(3600).unwrap().is_empty());
        // A freshly recorded one is recent.
        s.record_potential("KPI", "key performance indicator", 1.0)
            .unwrap();
        assert!(
            s.recent_potentials(3600)
                .unwrap()
                .contains(&("KPI".into(), "key performance indicator".into()))
        );
    }

    #[test]
    fn spell_correction_preserves_age_and_merges_counts() {
        let s = Store::open_in_memory().unwrap();
        s.record_potential("MVP", "minimum viabel product", 1.0)
            .unwrap();
        s.record_potential("MVP", "minimum viable product", 1.0)
            .unwrap();
        s.conn
            .execute(
                "UPDATE acronym_dictionary SET last_seen = datetime('now', '-2 hours')
                 WHERE source = 'mined'",
                [],
            )
            .unwrap();
        let words: std::collections::HashSet<String> = ["minimum", "viable", "product"]
            .iter()
            .map(|w| w.to_string())
            .collect();
        assert_eq!(s.spell_correct_potentials(&words).unwrap(), 1);
        let pots = s.potentials_for("MVP").unwrap();
        assert_eq!(pots.len(), 1);
        assert_eq!(
            (pots[0].0.as_str(), pots[0].1),
            ("minimum viable product", 2)
        );
        // Corrected, not re-sighted — it stays as prunable as it was.
        assert!(s.recent_potentials(3600).unwrap().is_empty());
    }

    #[test]
    fn consolidation_cadence_gates_on_last_run() {
        let s = Store::open_in_memory().unwrap();
        assert!(s.consolidate_due(3600).unwrap()); // never run → due
        s.mark_consolidated().unwrap();
        assert!(!s.consolidate_due(3600).unwrap()); // just ran → not due within an hour
        assert!(s.consolidate_due(0).unwrap()); // 0 interval → always due
    }

    #[test]
    fn pruning_spares_recently_seen_candidates() {
        let s = Store::open_in_memory().unwrap();
        s.record_candidate("XX").unwrap(); // seen just now
        assert_eq!(s.prune_noise_candidates(3600).unwrap(), 0); // within grace → kept
        assert_eq!(s.prune_noise_candidates(0).unwrap(), 1); // no grace → pruned
    }

    #[test]
    fn prune_drops_seen_once_candidates_with_nothing_mined() {
        let s = Store::open_in_memory().unwrap();
        s.record_candidate("XX").unwrap(); // seen once, no potentials → noise
        s.record_candidate("MVP").unwrap();
        s.record_potential("MVP", "minimum viable product", 1.0)
            .unwrap();
        assert_eq!(s.prune_noise_candidates(0).unwrap(), 1);
        let acrs: Vec<String> = s
            .candidates()
            .unwrap()
            .into_iter()
            .map(|(a, _)| a)
            .collect();
        assert!(acrs.contains(&"MVP".to_string()) && !acrs.contains(&"XX".to_string()));
    }

    #[test]
    fn prune_drops_non_acronym_candidates_however_often_seen() {
        let s = Store::open_in_memory().unwrap();
        let words: std::collections::HashSet<String> =
            ["where", "post"].iter().map(|w| w.to_string()).collect();
        for _ in 0..3 {
            s.record_candidate("WHERE").unwrap();
            s.record_candidate("V1").unwrap();
            s.record_candidate("ZQX").unwrap();
        }
        s.record_potential("WHERE", "we have every real error", 0.9)
            .unwrap();
        s.record_potential("ZQX", "zebra queue xray", 0.9).unwrap();
        s.declare_acronym("POST").unwrap(); // declared: kept though it's a word

        assert_eq!(s.prune_non_acronyms(Some(&words)).unwrap(), (2, 1));
        let acrs: Vec<String> = s
            .candidates()
            .unwrap()
            .into_iter()
            .map(|(a, _)| a)
            .collect();
        assert_eq!(acrs, ["ZQX", "POST"]);
        assert!(s.potentials_for("WHERE").unwrap().is_empty());
        assert_eq!(s.potentials_for("ZQX").unwrap().len(), 1);
    }

    #[test]
    fn without_a_wordlist_prune_keeps_word_candidates() {
        let s = Store::open_in_memory().unwrap();
        s.record_candidate("WHERE").unwrap();
        s.record_candidate("TODO").unwrap();
        assert_eq!(s.prune_non_acronyms(None).unwrap(), (1, 0));
        let acrs: Vec<String> = s
            .candidates()
            .unwrap()
            .into_iter()
            .map(|(a, _)| a)
            .collect();
        assert_eq!(acrs, ["WHERE"]);
    }

    #[test]
    fn ignoring_hides_an_acronym_but_keeps_its_expansions() {
        let s = Store::open_in_memory().unwrap();
        s.add_entry("API", "Application Programming Interface", "user")
            .unwrap();
        s.ignore_acronym("api").unwrap(); // case-insensitive
        // Excluded from the trie/mining source...
        assert!(!s.all_acronyms().unwrap().contains(&"API".to_string()));
        // ...but the confirmed expansion is still there (dormant), not deleted.
        assert_eq!(s.expansions_for("API").unwrap().len(), 1);
        assert_eq!(s.ignored_acronyms().unwrap(), vec!["API".to_string()]);
        // Reversible, and idempotent on the way back.
        assert!(s.unignore_acronym("API").unwrap());
        assert!(s.all_acronyms().unwrap().contains(&"API".to_string()));
        assert!(!s.unignore_acronym("API").unwrap()); // already not ignored
    }

    #[test]
    fn ignored_acronyms_are_inert_for_candidates_and_mining() {
        let s = Store::open_in_memory().unwrap();
        s.ignore_acronym("XX").unwrap();
        s.record_candidate("XX").unwrap(); // no-op while ignored
        s.record_potential("XX", "extra example", 1.0).unwrap(); // no-op
        assert!(s.candidates().unwrap().iter().all(|(a, _)| a != "XX"));
        assert!(s.potentials_for("XX").unwrap().is_empty());
        assert!(!s.watch_list().unwrap().contains(&"XX".to_string()));
        assert_eq!(s.ignored_count().unwrap(), 1);
    }

    #[test]
    fn ignoring_clears_existing_speculation_but_not_confirmed_rows() {
        let s = Store::open_in_memory().unwrap();
        s.add_entry("MVP", "Minimum Viable Product", "user")
            .unwrap();
        s.record_candidate("MVP").unwrap();
        s.record_potential("MVP", "most valuable player", 1.0)
            .unwrap();
        s.ignore_acronym("MVP").unwrap();
        // Speculative trail gone...
        assert!(s.potentials_for("MVP").unwrap().is_empty());
        assert!(s.candidates().unwrap().iter().all(|(a, _)| a != "MVP"));
        // ...confirmed expansion kept.
        assert_eq!(s.expansions_for("MVP").unwrap().len(), 1);
    }

    #[test]
    fn adding_an_expansion_unmutes_an_ignored_acronym() {
        let s = Store::open_in_memory().unwrap();
        s.ignore_acronym("MVP").unwrap();
        assert!(s.ignored_acronyms().unwrap().contains(&"MVP".to_string()));
        // An explicit definition reactivates it.
        s.add_entry("mvp", "Minimum Viable Product", "user")
            .unwrap();
        assert!(s.ignored_acronyms().unwrap().is_empty());
        assert!(s.all_acronyms().unwrap().contains(&"MVP".to_string()));
    }

    #[test]
    fn backup_snapshots_data_then_clear_all_wipes_it() {
        let s = Store::open_in_memory().unwrap();
        s.add_entry("MVP", "Minimum Viable Product", "user")
            .unwrap();
        s.record_candidate("XYZ").unwrap();
        s.ignore_acronym("API").unwrap();

        // Snapshot to a temp file.
        let dir = std::env::temp_dir().join(format!("ae-storetest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let backup = dir.join("snap.db");
        let _ = std::fs::remove_file(&backup);
        s.backup_to(&backup).unwrap();

        // Wipe clears every table and reports what it removed.
        let (entries, candidates) = s.clear_all().unwrap();
        assert!(entries >= 1 && candidates >= 1);
        assert_eq!(s.count().unwrap(), 0);
        assert!(s.candidates().unwrap().is_empty());
        assert!(s.ignored_acronyms().unwrap().is_empty());

        // The snapshot still holds the pre-wipe state.
        let restored = Store::open(&backup).unwrap();
        assert_eq!(restored.expansions_for("MVP").unwrap().len(), 1);
        assert!(
            restored
                .ignored_acronyms()
                .unwrap()
                .contains(&"API".to_string())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn context_embeddings_round_trip() {
        let s = Store::open_in_memory().unwrap();
        let id = s
            .add_entry("KPI", "Key Performance Indicator", "user")
            .unwrap();
        let v: Vec<f32> = (0..MRL_DIMS).map(|i| i as f32 / 10.0).collect();
        s.add_context(id, &v).unwrap();
        let back = s.contexts_for(id).unwrap();
        assert_eq!(back, vec![v]);
    }

    #[test]
    fn shares_round_to_two_decimals_and_need_a_denominator() {
        assert_eq!(share(7, 9), Some(0.78));
        assert_eq!(share(1, 3), Some(0.33));
        assert_eq!(share(0, 4), Some(0.0));
        assert_eq!(share(0, 0), None);
    }

    #[test]
    fn stats_tell_builtins_from_your_own_verified_rows() {
        let s = Store::open_in_memory().unwrap();
        s.seed_defaults().unwrap();
        s.delete_acronym("RAM").unwrap();
        s.add_entry("KPI", "Key Process Input", "user").unwrap();
        let st = s.stats().unwrap();
        assert_eq!(st.verified, DEFAULT_DICTIONARY.len());
        assert_eq!(st.builtin, DEFAULT_DICTIONARY.len() - 1);
        assert_eq!(st.acronyms, DEFAULT_DICTIONARY.len() - 1);
    }

    /// `(expansion, source)` rows of `acronym`, confirmed and mined alike.
    fn rows_of(s: &Store, acronym: &str) -> Vec<(String, String)> {
        let mut stmt = s
            .conn
            .prepare(
                "SELECT expansion, source FROM acronym_dictionary WHERE acronym = ?1 ORDER BY id",
            )
            .unwrap();
        stmt.query_map(params![acronym], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap()
    }

    #[test]
    fn add_treats_expansions_differing_only_in_case_as_one_row() {
        let row = |e: &str, src: &str| vec![(e.to_string(), src.to_string())];

        // Same strength: the existing row (and its casing) stands.
        let s = Store::open_in_memory().unwrap();
        let id = s.add_entry("BK", "Buildkite", "inline").unwrap();
        assert_eq!(s.add_entry("BK", "buildkite", "inline").unwrap(), id);
        assert_eq!(rows_of(&s, "BK"), row("Buildkite", "inline"));

        // A stronger source upgrades the row and brings its casing with it.
        assert_eq!(s.add_entry("BK", "BuildKite", "user").unwrap(), id);
        assert_eq!(rows_of(&s, "BK"), row("BuildKite", "user"));

        // A weaker source never rewrites a user row.
        s.add_entry("BK", "buildkite", "inline").unwrap();
        assert_eq!(rows_of(&s, "BK"), row("BuildKite", "user"));

        // A mined (lowercased) row is confirmed in place, not duplicated.
        let s = Store::open_in_memory().unwrap();
        s.record_potential("WQ", "widget quorum", 1.0).unwrap();
        s.add_entry("WQ", "Widget Quorum", "inline").unwrap();
        assert_eq!(rows_of(&s, "WQ"), row("Widget Quorum", "inline"));
    }

    #[test]
    fn add_folds_onto_the_confirmed_row_over_a_mined_case_sibling() {
        let row = |e: &str, src: &str| (e.to_string(), src.to_string());

        // Re-adding with the mined row's casing must not confirm the mined row
        // alongside the user one.
        let s = Store::open_in_memory().unwrap();
        let id = s.add_entry("BK", "Buildkite", "user").unwrap();
        s.record_potential("BK", "buildkite", 1.0).unwrap();
        assert_eq!(s.add_entry("BK", "buildkite", "user").unwrap(), id);
        assert_eq!(
            s.expansions_for("BK").unwrap(),
            vec![(id, "Buildkite".to_string(), "user".to_string())]
        );

        // Upgrading the confirmed row to the mined row's casing absorbs the
        // mined row instead of colliding with it.
        let s = Store::open_in_memory().unwrap();
        let id = s.add_entry("BK", "Buildkite", "inline").unwrap();
        s.record_potential("BK", "buildkite", 1.0).unwrap();
        s.add_context(id, &[0.0; MRL_DIMS]).unwrap();
        assert_eq!(s.add_entry("BK", "buildkite", "user").unwrap(), id);
        assert_eq!(rows_of(&s, "BK"), vec![row("buildkite", "user")]);
        assert_eq!(s.contexts_for(id).unwrap().len(), 1);
    }

    #[test]
    fn add_prefers_the_exact_casing_among_legacy_case_siblings() {
        let s = Store::open_in_memory().unwrap();
        // Case-only siblings written before add folded case.
        s.conn
            .execute_batch(
                "INSERT INTO acronym_dictionary (acronym, expansion, source)
                 VALUES ('BK', 'Buildkite', 'inline'), ('BK', 'buildkite', 'inline');",
            )
            .unwrap();
        s.add_entry("BK", "buildkite", "user").unwrap();
        assert_eq!(
            rows_of(&s, "BK"),
            vec![
                ("Buildkite".to_string(), "inline".to_string()),
                ("buildkite".to_string(), "user".to_string()),
            ]
        );
    }

    #[test]
    fn delete_unverified_keeps_only_user_rows() {
        let s = Store::open_in_memory().unwrap();
        let v: Vec<f32> = vec![0.0; MRL_DIMS];
        let kept = s.add_entry("ZQ", "Zone Quota", "user").unwrap();
        let gone = s.add_entry("ZQ", "Zebra Queue", "inline").unwrap();
        s.add_context(kept, &v).unwrap();
        s.add_context(gone, &v).unwrap();
        s.record_potential("ZQ", "zany quokka", 1.0).unwrap();
        s.add_entry("WQ", "Widget Quorum", "inline").unwrap();

        // Scoped: inline and mined rows of ZQ go; ZQ's user row and WQ stay.
        assert_eq!(s.delete_unverified(Some("zq")).unwrap(), 2);
        let zq: Vec<String> = s
            .expansions_for("ZQ")
            .unwrap()
            .into_iter()
            .map(|r| r.1)
            .collect();
        assert_eq!(zq, vec!["Zone Quota".to_string()]);
        assert!(s.potentials_for("ZQ").unwrap().is_empty());
        assert_eq!(s.contexts_for(kept).unwrap().len(), 1);
        assert!(s.contexts_for(gone).unwrap().is_empty());
        assert_eq!(s.expansions_for("WQ").unwrap().len(), 1);

        // Unscoped: every acronym.
        assert_eq!(s.delete_unverified(None).unwrap(), 1);
        assert!(s.expansions_for("WQ").unwrap().is_empty());
        assert_eq!(s.expansions_for("ZQ").unwrap().len(), 1);
    }

    /// Insert rows verbatim, bypassing add_entry's case fold — the shapes a
    /// database written by an older `ae` holds.
    fn plant(s: &Store, rows: &[(&str, &str, &str)]) -> Vec<i64> {
        rows.iter()
            .map(|(acronym, expansion, source)| {
                s.conn
                    .execute(
                        "INSERT INTO acronym_dictionary (acronym, expansion, source)
                         VALUES (?1, ?2, ?3)",
                        params![acronym, expansion, source],
                    )
                    .unwrap();
                s.conn.last_insert_rowid()
            })
            .collect()
    }

    #[test]
    fn prune_drops_unaligned_inline_rows_and_keeps_the_rest() {
        let s = Store::open_in_memory().unwrap();
        let ids = plant(
            &s,
            &[
                ("WQ", "Widget Quorum", "inline"), // aligned: kept
                ("XY", "Red Balloon", "inline"),   // unaligned: dropped
                ("ZQ", "Red Balloon", "user"),     // yours, aligned or not
                ("WS", "WidgetSoft", "inline"),    // camel case spells it: kept
            ],
        );
        for id in &ids {
            s.add_context(*id, &[0.0; MRL_DIMS]).unwrap();
        }
        assert_eq!(s.stale_confirmed_count().unwrap(), 1);

        let stats = s.prune(PRUNE_MIN_CONFIDENCE, 0).unwrap();
        assert_eq!((stats.unaligned, stats.case_merged), (1, 0));
        assert!(s.expansions_for("XY").unwrap().is_empty());
        assert!(s.contexts_for(ids[1]).unwrap().is_empty());
        assert_eq!(s.expansions_for("WQ").unwrap().len(), 1);
        assert_eq!(s.expansions_for("ZQ").unwrap().len(), 1);
        assert_eq!(s.expansions_for("WS").unwrap().len(), 1);
        assert_eq!(s.stale_confirmed_count().unwrap(), 0);
    }

    #[test]
    fn prune_merges_confirmed_case_duplicates_by_the_add_rule() {
        let s = Store::open_in_memory().unwrap();
        let ids = plant(
            &s,
            &[
                // Yours wins over an inline sibling, casing included.
                ("AST", "Abstract Syntax Tree", "inline"),
                ("AST", "abstract syntax tree", "user"),
                // Equal sources: the older row's casing stands.
                ("WQ", "Widget Quorum", "inline"),
                ("WQ", "widget quorum", "inline"),
                // A mined sibling is speculation, not a duplicate to merge.
                ("ZQ", "Zone Quota", "user"),
                ("ZQ", "zone quota", "mined"),
            ],
        );
        s.add_context(ids[0], &[0.0; MRL_DIMS]).unwrap();
        s.add_context(ids[3], &[0.0; MRL_DIMS]).unwrap();
        assert_eq!(s.stale_confirmed_count().unwrap(), 2);

        let stats = s.prune(PRUNE_MIN_CONFIDENCE, 0).unwrap();
        assert_eq!((stats.unaligned, stats.case_merged), (0, 2));
        let shown = |a: &str| s.expansions_for(a).unwrap();
        assert_eq!(
            shown("AST"),
            vec![(ids[1], "abstract syntax tree".into(), "user".into())]
        );
        assert_eq!(
            shown("WQ"),
            vec![(ids[2], "Widget Quorum".into(), "inline".into())]
        );
        // The dropped rows' contexts follow the survivor.
        assert_eq!(s.contexts_for(ids[1]).unwrap().len(), 1);
        assert_eq!(s.contexts_for(ids[2]).unwrap().len(), 1);
        assert_eq!(shown("ZQ").len(), 1);
    }

    #[test]
    fn the_periodic_consolidation_leaves_confirmed_rows_alone() {
        // Deleting dictionary entries takes a backup, which only `ae prune` does.
        let s = Store::open_in_memory().unwrap();
        plant(
            &s,
            &[
                ("XY", "Red Balloon", "inline"),
                ("WQ", "Widget Quorum", "inline"),
                ("WQ", "widget quorum", "inline"),
            ],
        );
        s.consolidate(PRUNE_MIN_CONFIDENCE, 0).unwrap();
        assert_eq!(s.stale_confirmed_count().unwrap(), 2);
    }
}
