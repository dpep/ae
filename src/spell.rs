//! Dictionary spell-correction for mined expansions, against the *system* word
//! list — no bundle, no download. If no list is present we simply skip it.
//!
//! Used only by `prune` over speculative (mined) expansions, so the blast radius
//! of a wrong correction is limited to suggestions a human will vet before
//! promoting. Words absent from the list (jargon, proper nouns) are left alone
//! unless an edit-distance-1 neighbour exists in it.

use std::collections::HashSet;

const WORDLIST_PATHS: &[&str] = &["/usr/share/dict/words", "/usr/dict/words"];

/// Load the system word list (lowercased), or `None` if none is installed.
pub fn load_wordlist() -> Option<HashSet<String>> {
    for path in WORDLIST_PATHS {
        if let Ok(text) = std::fs::read_to_string(path) {
            let words: HashSet<String> = text
                .lines()
                .map(|w| w.trim().to_lowercase())
                .filter(|w| !w.is_empty())
                .collect();
            if !words.is_empty() {
                return Some(words);
            }
        }
    }
    None
}

/// Correct an expansion word by word: any word not in `words` is replaced by an
/// edit-distance-1 neighbour that is, if one exists. Short words (`< 4` chars,
/// e.g. fillers) are left alone.
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
    edits1(&lower)
        .into_iter()
        .find(|c| words.contains(c))
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
    fn leaves_unknown_words_without_a_neighbour_alone() {
        let w = words(&["product"]);
        assert_eq!(correct("kubernetes product", &w), "kubernetes product");
    }
}
