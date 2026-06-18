//! Embedded SQLite storage: the acronym dictionary and the 64-d context store.
//!
//! Vectors are stored as raw little-endian `f32` BLOBs rather than in a
//! `sqlite-vec` `vec0` virtual table — the bundled SQLite doesn't carry that
//! extension, and at this corpus size cosine ranking in Rust over the candidate
//! set is simpler and fast. The API hides the representation, so swapping in
//! `vec0` later is a storage-layer change only. See docs/ROADMAP.md.

use std::path::Path;

use rusqlite::{Connection, Result, params};

use crate::mrl::MRL_DIMS;

const SCHEMA: &str = "
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS acronym_dictionary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    acronym TEXT NOT NULL,
    expansion TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_acronym_lookup
    ON acronym_dictionary(acronym, expansion);

CREATE TABLE IF NOT EXISTS acronym_contexts (
    acronym_id INTEGER NOT NULL REFERENCES acronym_dictionary(id),
    context_embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_context_acronym
    ON acronym_contexts(acronym_id);

-- Acronym-shaped tokens seen but not (yet) defined, with how often.
CREATE TABLE IF NOT EXISTS candidate_acronyms (
    acronym TEXT PRIMARY KEY,
    count INTEGER NOT NULL DEFAULT 1,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Speculative expansions for candidates: phrases whose word-initials spell the
-- acronym, mined from text that mentions it. Recurrence drives confidence.
CREATE TABLE IF NOT EXISTS potential_expansions (
    acronym TEXT NOT NULL,
    expansion TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 1,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (acronym, expansion)
);
";

/// A small built-in dictionary so expansion works on a fresh database.
pub const DEFAULT_DICTIONARY: &[(&str, &str)] = &[
    ("OKR", "Objectives and Key Results"),
    ("KPI", "Key Performance Indicator"),
    ("API", "Application Programming Interface"),
    ("CLI", "Command Line Interface"),
    ("UDS", "Unix Domain Socket"),
    ("MRL", "Matryoshka Representation Learning"),
    ("LLM", "Large Language Model"),
    ("RAM", "Random Access Memory"),
];

pub struct Store {
    conn: Connection,
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
        Ok(Self { conn })
    }

    /// Insert an `(acronym, expansion)` pair, returning its row id. Acronyms are
    /// stored uppercased. Idempotent: an existing pair returns its current id.
    pub fn add_entry(&self, acronym: &str, expansion: &str) -> Result<i64> {
        let acronym = acronym.trim().to_uppercase();
        let expansion = expansion.trim();
        self.conn.execute(
            "INSERT OR IGNORE INTO acronym_dictionary (acronym, expansion) VALUES (?1, ?2)",
            params![acronym, expansion],
        )?;
        // It's now defined, so it's no longer an open candidate.
        self.clear_candidate(&acronym)?;
        self.conn.query_row(
            "SELECT id FROM acronym_dictionary WHERE acronym = ?1 AND expansion = ?2",
            params![acronym, expansion],
            |row| row.get(0),
        )
    }

    /// Record one sighting of an undefined acronym-shaped token, bumping its
    /// occurrence count.
    pub fn record_candidate(&self, acronym: &str) -> Result<()> {
        let acronym = acronym.trim().to_uppercase();
        if acronym.is_empty() {
            return Ok(());
        }
        self.conn.execute(
            "INSERT INTO candidate_acronyms (acronym, count) VALUES (?1, 1)
             ON CONFLICT(acronym) DO UPDATE SET count = count + 1, last_seen = CURRENT_TIMESTAMP",
            params![acronym],
        )?;
        Ok(())
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
        self.conn.execute(
            "DELETE FROM potential_expansions WHERE acronym = ?1",
            params![acronym],
        )?;
        Ok(())
    }

    /// Record one sighting of a speculative expansion for a candidate acronym.
    pub fn record_potential(&self, acronym: &str, expansion: &str) -> Result<()> {
        let acronym = acronym.trim().to_uppercase();
        let expansion = expansion.trim().to_lowercase();
        if acronym.is_empty() || expansion.is_empty() {
            return Ok(());
        }
        self.conn.execute(
            "INSERT INTO potential_expansions (acronym, expansion, count) VALUES (?1, ?2, 1)
             ON CONFLICT(acronym, expansion) DO UPDATE SET count = count + 1, last_seen = CURRENT_TIMESTAMP",
            params![acronym, expansion],
        )?;
        Ok(())
    }

    /// Speculative `(expansion, count)` pairs for one acronym, most-seen first.
    pub fn potentials_for(&self, acronym: &str) -> Result<Vec<(String, i64)>> {
        let acronym = acronym.trim().to_uppercase();
        let mut stmt = self.conn.prepare(
            "SELECT expansion, count FROM potential_expansions
             WHERE acronym = ?1 ORDER BY count DESC, expansion",
        )?;
        let rows = stmt
            .query_map(params![acronym], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// All speculative `(acronym, expansion, count)` triples, for `suggest`.
    pub fn all_potentials(&self) -> Result<Vec<(String, String, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT acronym, expansion, count FROM potential_expansions
             ORDER BY acronym, count DESC, expansion",
        )?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Seed the built-in dictionary if the table is empty. Returns the number of
    /// rows inserted.
    pub fn seed_defaults(&self) -> Result<usize> {
        if self.count()? > 0 {
            return Ok(0);
        }
        let mut n = 0;
        for (acr, exp) in DEFAULT_DICTIONARY {
            self.add_entry(acr, exp)?;
            n += 1;
        }
        Ok(n)
    }

    /// All `(id, expansion)` rows for `acronym` (case-insensitive).
    pub fn expansions_for(&self, acronym: &str) -> Result<Vec<(i64, String)>> {
        let acronym = acronym.trim().to_uppercase();
        let mut stmt = self.conn.prepare(
            "SELECT id, expansion FROM acronym_dictionary WHERE acronym = ?1 ORDER BY id",
        )?;
        let rows = stmt
            .query_map(params![acronym], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Every distinct acronym — used to hydrate the trie.
    pub fn all_acronyms(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT acronym FROM acronym_dictionary ORDER BY acronym")?;
        let rows = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Every `(acronym, expansion)` pair, ordered — for `list`.
    pub fn all_entries(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT acronym, expansion FROM acronym_dictionary ORDER BY acronym, expansion",
        )?;
        let rows = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Entries whose acronym or expansion contains `query` (case-insensitive).
    pub fn search(&self, query: &str) -> Result<Vec<(String, String)>> {
        let pattern = format!("%{}%", query.trim());
        let mut stmt = self.conn.prepare(
            "SELECT acronym, expansion FROM acronym_dictionary
             WHERE acronym LIKE ?1 OR expansion LIKE ?1
             ORDER BY acronym, expansion",
        )?;
        let rows = stmt
            .query_map(params![pattern], |row| Ok((row.get(0)?, row.get(1)?)))?
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

    pub fn count(&self) -> Result<i64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM acronym_dictionary", [], |row| {
                row.get(0)
            })
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
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_look_up_an_entry() {
        let s = Store::open_in_memory().unwrap();
        let id = s.add_entry("kpi", "Key Performance Indicator").unwrap();
        let rows = s.expansions_for("KPI").unwrap();
        assert_eq!(rows, vec![(id, "Key Performance Indicator".to_string())]);
    }

    #[test]
    fn duplicate_pairs_are_idempotent() {
        let s = Store::open_in_memory().unwrap();
        let a = s.add_entry("KPI", "Key Performance Indicator").unwrap();
        let b = s.add_entry("KPI", "Key Performance Indicator").unwrap();
        assert_eq!(a, b);
        assert_eq!(s.count().unwrap(), 1);
    }

    #[test]
    fn one_acronym_can_have_several_expansions() {
        let s = Store::open_in_memory().unwrap();
        s.add_entry("PT", "Physical Therapy").unwrap();
        s.add_entry("PT", "Part Time").unwrap();
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
        s.add_entry("KPI", "Key Performance Indicator").unwrap();
        s.add_entry("OKR", "Objectives and Key Results").unwrap();
        // Matches on the expansion text ("Key" appears in both).
        assert_eq!(s.search("key").unwrap().len(), 2);
        // Matches on the acronym.
        assert_eq!(
            s.search("kpi").unwrap(),
            vec![("KPI".into(), "Key Performance Indicator".into())]
        );
        assert!(s.search("nope").unwrap().is_empty());
    }

    #[test]
    fn delete_removes_entries_and_is_counted() {
        let s = Store::open_in_memory().unwrap();
        s.add_entry("PT", "Physical Therapy").unwrap();
        s.add_entry("PT", "Part Time").unwrap();
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
        s.add_entry("MVP", "Minimum Viable Product").unwrap();
        assert_eq!(s.candidates().unwrap(), vec![("ABC".into(), 1)]);
    }

    #[test]
    fn context_embeddings_round_trip() {
        let s = Store::open_in_memory().unwrap();
        let id = s.add_entry("KPI", "Key Performance Indicator").unwrap();
        let v: Vec<f32> = (0..MRL_DIMS).map(|i| i as f32 / 10.0).collect();
        s.add_context(id, &v).unwrap();
        let back = s.contexts_for(id).unwrap();
        assert_eq!(back, vec![v]);
    }
}
