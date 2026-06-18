//! A thread-safe prefix tree for fast "is this token a known acronym?" checks.
//!
//! The Leader holds one [`SharedTrie`] warm in memory and reads it under a
//! shared lock for every expansion query, while occasional dictionary growth
//! takes the write lock. Acronyms are matched case-insensitively by storing
//! their uppercased form.

use std::collections::HashMap;
use std::sync::RwLock;

/// One node in the prefix tree. A node whose `is_acronym` is set marks the end
/// of a stored acronym.
#[derive(Default, Debug)]
pub struct TrieNode {
    pub children: HashMap<char, TrieNode>,
    pub is_acronym: bool,
}

/// A trie behind an `RwLock` so the warm Leader can share reads across
/// connections while inserts are serialized.
#[derive(Default)]
pub struct SharedTrie {
    pub root: RwLock<TrieNode>,
}

impl SharedTrie {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an acronym (case-insensitively). Empty input is ignored.
    pub fn insert(&self, acronym: &str) {
        let key = acronym.trim().to_uppercase();
        if key.is_empty() {
            return;
        }
        let mut node = self.root.write().unwrap();
        let mut cur = &mut *node;
        for ch in key.chars() {
            cur = cur.children.entry(ch).or_default();
        }
        cur.is_acronym = true;
    }

    /// True if `acronym` was inserted (case-insensitive exact match).
    pub fn contains(&self, acronym: &str) -> bool {
        let key = acronym.trim().to_uppercase();
        if key.is_empty() {
            return false;
        }
        let node = self.root.read().unwrap();
        let mut cur = &*node;
        for ch in key.chars() {
            match cur.children.get(&ch) {
                Some(next) => cur = next,
                None => return false,
            }
        }
        cur.is_acronym
    }

    /// Number of acronyms stored.
    pub fn len(&self) -> usize {
        fn count(node: &TrieNode) -> usize {
            let here = usize::from(node.is_acronym);
            here + node.children.values().map(count).sum::<usize>()
        }
        count(&self.root.read().unwrap())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_then_contains() {
        let t = SharedTrie::new();
        t.insert("KPI");
        assert!(t.contains("KPI"));
        assert!(!t.contains("OKR"));
    }

    #[test]
    fn matching_is_case_insensitive() {
        let t = SharedTrie::new();
        t.insert("kpi");
        assert!(t.contains("KPI"));
        assert!(t.contains("Kpi"));
    }

    #[test]
    fn a_prefix_of_a_stored_acronym_is_not_a_match() {
        let t = SharedTrie::new();
        t.insert("KPIS");
        assert!(!t.contains("KPI"));
        assert!(t.contains("KPIS"));
    }

    #[test]
    fn empty_input_is_ignored() {
        let t = SharedTrie::new();
        t.insert("   ");
        assert!(t.is_empty());
        assert!(!t.contains(""));
    }

    #[test]
    fn len_counts_distinct_acronyms() {
        let t = SharedTrie::new();
        t.insert("KPI");
        t.insert("OKR");
        t.insert("KPI"); // duplicate
        assert_eq!(t.len(), 2);
    }
}
