//! Acronym-shaped tokens that aren't acronyms. Shared by candidate detection
//! (the engine) and `prune` (the store), so a token rejected in new text is
//! also cleared from what was recorded before the rule existed.

use std::collections::HashSet;

/// Acronym-shaped tokens that are clearly not acronyms. Applied only to
/// candidates: a dictionary entry is resolved before this runs, so `ae add CAT`
/// still expands. `words` is [`crate::spell::system_short_words`], not the full
/// list, whose lowercased proper nouns would make `ADA` a word.
pub fn is_noise(token: &str, words: Option<&HashSet<String>>) -> bool {
    !COMMON_ACRONYMS.contains(&token)
        && (is_numbered_label(token)
            || NOISE_TOKENS.contains(&token)
            || words.is_some_and(|w| is_word(&token.to_lowercase(), w)))
}

/// Real acronyms whose dictionary sense is too obscure to be what someone
/// shouting means — a sloth, a fish, a Greek letter. Not `IT`: "it" is
/// everyday English. Keep this short; `ae add` declares anything else.
const COMMON_ACRONYMS: &[&str] = &["AI", "IDE", "ISO", "ROI", "ETA", "SLA"];

/// `lower` is in the wordlist, or inflects a word in it: the list has base
/// forms only, so `failed` and `values` need `fail` and `value`. The stem must
/// keep four letters — `aws`/`cors`/`saas` would otherwise strip to the words
/// `aw`/`cor`/`saa`. (`-ing` and doubled consonants can't leave four letters
/// within the six-letter acronym shape, so they aren't stripped.)
fn is_word(lower: &str, words: &HashSet<String>) -> bool {
    const MIN_STEM: usize = 4;
    words.contains(lower)
        || ["s", "es", "ed"].iter().any(|suffix| {
            lower
                .strip_suffix(suffix)
                .is_some_and(|stem| stem.len() >= MIN_STEM && words.contains(stem))
        })
        // moved → move
        || lower
            .strip_suffix('d')
            .is_some_and(|stem| stem.ends_with('e') && stem.len() >= MIN_STEM && words.contains(stem))
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

    fn word_set(list: &[&str]) -> HashSet<String> {
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
    fn an_inflected_word_is_noise_though_the_wordlist_has_only_its_base() {
        let words = word_set(&["fail", "value", "move", "merge", "pass"]);
        for token in ["FAILED", "VALUES", "MOVED", "MERGES", "PASSES"] {
            assert!(is_noise(token, Some(&words)), "{token} should be noise");
        }
    }

    #[test]
    fn an_acronym_ending_in_s_survives_a_short_stem_that_is_a_word() {
        // aw, cor, saa, io, ad are all in the system wordlist.
        let words = word_set(&["aw", "cor", "saa", "io", "ad", "kpi"]);
        for token in ["AWS", "CORS", "SAAS", "IOS", "ADS"] {
            assert!(!is_noise(token, Some(&words)), "{token} is a real acronym");
        }
    }

    #[test]
    fn common_acronyms_with_obscure_word_senses_stay_candidates() {
        let words = word_set(&["ai", "ide", "roi", "iso", "eta", "sla", "it"]);
        for token in ["AI", "IDE", "ROI", "ISO", "ETA", "SLA"] {
            assert!(!is_noise(token, Some(&words)), "{token} is a real acronym");
        }
        // "it" is everyday English; shouted text would flood the list with it.
        assert!(is_noise("IT", Some(&words)));
    }
}
