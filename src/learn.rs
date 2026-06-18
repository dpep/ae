//! Stage 2: rule-based learning — extract acronyms *defined inline* in text.
//!
//! Two deterministic structural patterns capture the overwhelming majority of
//! inline definitions:
//!
//! - **Alpha** — `ACR (Definition Words)`, e.g. `KPI (Key Performance Indicator)`
//! - **Beta**  — `Definition Words (ACR)`, e.g. `Key Performance Indicator (KPI)`
//!
//! When the definition's word initials spell the acronym we're confident the
//! pair is real (0.95); otherwise it's a weaker structural guess (0.6). Beta's
//! definition capture is greedy, so we realign it to the tightest trailing
//! window whose initials match before trusting it.

use std::sync::LazyLock;

use regex::Regex;

use crate::types::Extraction;

static ALPHA: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?P<acronym>[A-Z]{2,6})\s\((?P<definition>[A-Za-z\s]{4,60})\)").unwrap()
});
static BETA: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?P<definition>[A-Za-z\s]{4,60})\s\((?P<acronym>[A-Z]{2,6})\)").unwrap()
});

const STRONG: f32 = 0.95;
const WEAK: f32 = 0.6;

/// Extract every inline acronym definition in `text`, de-duplicated by
/// `(acronym, definition)`.
pub fn extract(text: &str) -> Vec<Extraction> {
    let mut out: Vec<Extraction> = Vec::new();

    for caps in ALPHA.captures_iter(text) {
        let acronym = caps["acronym"].to_string();
        let raw = caps["definition"].trim();
        push_unique(&mut out, candidate(acronym, raw, "alpha"));
    }
    for caps in BETA.captures_iter(text) {
        let acronym = caps["acronym"].to_string();
        let raw = caps["definition"].trim();
        push_unique(&mut out, candidate(acronym, raw, "beta"));
    }
    out
}

/// Build a candidate, realigning the definition to the acronym's initials when
/// possible and scoring confidence by whether that alignment holds.
fn candidate(acronym: String, raw_definition: &str, pattern_type: &str) -> Extraction {
    let (definition, aligned) = realign(raw_definition, &acronym);
    Extraction {
        acronym,
        extracted_definition: definition,
        pattern_type: pattern_type.to_string(),
        confidence: if aligned { STRONG } else { WEAK },
    }
}

/// Realign a (possibly noisy) definition to the acronym it defines.
///
/// The definition sits right before the parenthetical, so its final word should
/// map to the acronym's last letter. We look for the tightest trailing window
/// whose word-initials *contain the acronym letters as a subsequence* — which
/// tolerates skipped function words (`OKR` = **O**bjectives and **K**ey
/// **R**esults) and trims leading noise (`Track Objectives…`). Returns
/// `(definition, aligned)`: the trimmed window and `true` on a hit, otherwise
/// the whole trimmed phrase and `false`.
fn realign(definition: &str, acronym: &str) -> (String, bool) {
    let words: Vec<&str> = definition.split_whitespace().collect();
    let inits: Vec<char> = words.iter().map(|w| initial(w)).collect();
    let target: Vec<char> = acronym.chars().map(|c| c.to_ascii_uppercase()).collect();

    // The window must end at the acronym's last letter — anchor at the tail.
    let ends_ok = inits.last() == target.last();
    if ends_ok {
        // Largest start (tightest window) whose first word opens the acronym and
        // whose initials still spell it as a subsequence.
        for start in (0..words.len()).rev() {
            if inits[start] == target[0] && is_subsequence(&target, &inits[start..]) {
                return (words[start..].join(" "), true);
            }
        }
    }
    (words.join(" "), false)
}

/// The uppercased first alphanumeric letter of a word (`'?'` if it has none).
fn initial(word: &str) -> char {
    word.chars()
        .find(|c| c.is_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .unwrap_or('?')
}

/// Is `needle` an in-order subsequence of `haystack`?
fn is_subsequence(needle: &[char], haystack: &[char]) -> bool {
    let mut it = haystack.iter();
    needle.iter().all(|n| it.any(|h| h == n))
}

fn push_unique(out: &mut Vec<Extraction>, c: Extraction) {
    let dup = out.iter().any(|e| {
        e.acronym.eq_ignore_ascii_case(&c.acronym)
            && e.extracted_definition
                .eq_ignore_ascii_case(&c.extracted_definition)
    });
    if !dup {
        out.push(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_alpha_extracts_acronym_first() {
        let got = extract("Our KPI (Key Performance Indicator) matters.");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].acronym, "KPI");
        assert_eq!(got[0].extracted_definition, "Key Performance Indicator");
        assert_eq!(got[0].pattern_type, "alpha");
        assert_eq!(got[0].confidence, STRONG);
    }

    #[test]
    fn pattern_beta_extracts_definition_first() {
        let got = extract("Track Objectives and Key Results (OKR) quarterly.");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].acronym, "OKR");
        assert_eq!(got[0].extracted_definition, "Objectives and Key Results");
        assert_eq!(got[0].pattern_type, "beta");
    }

    #[test]
    fn beta_realigns_past_leading_noise() {
        // The greedy capture grabs "Our quarterly Key Performance Indicator";
        // realignment trims it to the three words whose initials spell KPI.
        let got = extract("Our quarterly Key Performance Indicator (KPI) review.");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].extracted_definition, "Key Performance Indicator");
        assert_eq!(got[0].confidence, STRONG);
    }

    #[test]
    fn unaligned_definition_gets_weak_confidence() {
        // Initials (R, B) don't spell XY, so it's a weak structural guess.
        let got = extract("Red Balloon (XY) floats.");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].confidence, WEAK);
    }

    #[test]
    fn plain_text_yields_nothing() {
        assert!(extract("A sentence with no inline definitions at all.").is_empty());
    }

    #[test]
    fn duplicate_definitions_are_collapsed() {
        let got =
            extract("KPI (Key Performance Indicator). Again: KPI (Key Performance Indicator).");
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn both_patterns_in_one_text() {
        let got = extract("KPI (Key Performance Indicator) and Objectives and Key Results (OKR).");
        let acros: Vec<&str> = got.iter().map(|c| c.acronym.as_str()).collect();
        assert!(acros.contains(&"KPI"));
        assert!(acros.contains(&"OKR"));
    }
}
