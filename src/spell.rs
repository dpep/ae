//! The *system* word list — no bundle, no download. If no list is present,
//! everything that uses it is simply skipped.
//!
//! Two uses: `prune` spell-corrects speculative (mined) expansions against it,
//! so the blast radius of a wrong correction is limited to suggestions a human
//! will vet before promoting; and candidate tracking drops all-caps tokens that
//! are just ordinary words ([`short_words`]). Words absent from the list
//! (jargon, proper nouns) are left alone unless an edit-distance-1 neighbour
//! exists in it.

use std::collections::HashSet;
use std::sync::OnceLock;

const WORDLIST_PATHS: &[&str] = &["/usr/share/dict/words", "/usr/dict/words"];

/// Longest word kept by [`short_words`] — mirrors the acronym-shape cap in
/// `engine::is_acronym_shaped`; a longer word can never collide with one.
const SHORT_WORD_MAX: usize = 6;

fn read_wordlist() -> Option<String> {
    WORDLIST_PATHS.iter().find_map(|path| {
        std::fs::read_to_string(path)
            .ok()
            .filter(|text| !text.trim().is_empty())
    })
}

/// Load the system word list (lowercased), or `None` if none is installed.
pub fn load_wordlist() -> Option<HashSet<String>> {
    let words: HashSet<String> = read_wordlist()?
        .lines()
        .map(|w| w.trim().to_lowercase())
        .filter(|w| !w.is_empty())
        .collect();
    (!words.is_empty()).then_some(words)
}

/// Ordinary words short enough to be mistaken for an acronym, loaded once per
/// process — `None` when no system list is installed.
pub fn system_short_words() -> Option<&'static HashSet<String>> {
    static WORDS: OnceLock<Option<HashSet<String>>> = OnceLock::new();
    WORDS
        .get_or_init(|| Some(short_words(&read_wordlist()?)).filter(|w| !w.is_empty()))
        .as_ref()
}

/// The 2..=[`SHORT_WORD_MAX`]-letter entries written in lowercase. A
/// capitalized entry is a proper noun (`Ada`), not a word someone would shout.
fn short_words(text: &str) -> HashSet<String> {
    text.lines()
        .map(str::trim)
        .filter(|w| (2..=SHORT_WORD_MAX).contains(&w.len()))
        .filter(|w| w.bytes().all(|b| b.is_ascii_lowercase()))
        .map(str::to_string)
        .collect()
}

/// Correct an expansion word by word: any word not in `words` is replaced by an
/// edit-distance-1 neighbour that is, if one exists. Short words (`< 4` chars,
/// e.g. fillers) are left alone, and a correction never changes a word's
/// initial — the initials are what spell the acronym.
pub fn correct(expansion: &str, words: &HashSet<String>) -> String {
    expansion
        .split_whitespace()
        .map(|w| correct_word(w, words))
        .collect::<Vec<_>>()
        .join(" ")
}

fn correct_word(word: &str, words: &HashSet<String>) -> String {
    let lower = word.to_lowercase();
    if lower.chars().count() < 4 || words.contains(&lower) {
        return word.to_string();
    }
    let initial = lower.chars().next();
    edits1(&lower)
        .into_iter()
        .find(|c| c.chars().next() == initial && words.contains(c))
        .unwrap_or_else(|| word.to_string())
}

/// All strings one edit (delete / transpose / substitute / insert) from `word`.
fn edits1(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let mut out = Vec::new();
    for i in 0..chars.len() {
        let mut s = chars.clone();
        s.remove(i);
        out.push(s.into_iter().collect());
    }
    for i in 0..chars.len().saturating_sub(1) {
        let mut s = chars.clone();
        s.swap(i, i + 1);
        out.push(s.into_iter().collect());
    }
    for i in 0..chars.len() {
        for a in 'a'..='z' {
            let mut s = chars.clone();
            s[i] = a;
            out.push(s.into_iter().collect());
        }
    }
    for i in 0..=chars.len() {
        for a in 'a'..='z' {
            let mut s = chars.clone();
            s.insert(i, a);
            out.push(s.into_iter().collect());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(list: &[&str]) -> HashSet<String> {
        list.iter().map(|w| w.to_string()).collect()
    }

    #[test]
    fn corrects_a_one_edit_typo() {
        let w = words(&["minimum", "viable", "product"]);
        assert_eq!(
            correct("minimum viabel product", &w),
            "minimum viable product"
        );
    }

    #[test]
    fn leaves_known_and_short_words_alone() {
        let w = words(&["viable", "and"]);
        // "and" is short; "viable" is known; "qx" has no neighbour → unchanged.
        assert_eq!(correct("viable and qx", &w), "viable and qx");
    }

    #[test]
    fn never_changes_a_words_initial() {
        // "boding" is one substitution from "coding", but it would stop the
        // phrase spelling its acronym.
        let w = words(&["boding", "oding", "sessions"]);
        assert_eq!(correct("coding sessions", &w), "coding sessions");
    }

    #[test]
    fn leaves_unknown_words_without_a_neighbour_alone() {
        let w = words(&["product"]);
        assert_eq!(correct("kubernetes product", &w), "kubernetes product");
    }

    #[test]
    fn short_words_keep_only_short_lowercase_entries() {
        // Capitalized entries are proper nouns, not ordinary words; possessives
        // and long words can never collide with an acronym-shaped token.
        let w = short_words("cat\nAda\nwhere\ncat's\nextraordinary\na\n");
        assert_eq!(w, words(&["cat", "where"]));
    }
}
