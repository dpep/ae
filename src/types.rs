//! The serialized contract between the engine and its consumers.
//!
//! Every role (in-process CLI, Leader server, Follower proxy) ultimately
//! produces an [`AnalysisPayload`]. Field names here are stable — consumers
//! parse them, so renames are breaking changes.

use serde::{Deserialize, Serialize};

/// One candidate expansion for a known acronym, with a confidence in `[0, 1]`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct MatchCandidate {
    pub expansion: String,
    pub confidence: f32,
}

/// A known acronym found in the input, with its ranked candidate expansions.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ExpansionResult {
    pub acronym: String,
    /// The token as it appeared in the source text.
    pub text_slice: String,
    pub matches: Vec<MatchCandidate>,
}

/// An acronym/definition pair extracted from inline structure in the text
/// (e.g. `KPI (Key Performance Indicator)`).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Extraction {
    pub acronym: String,
    pub extracted_definition: String,
    /// Which rule matched — e.g. `"alpha"` or `"beta"`.
    pub pattern_type: String,
    pub confidence: f32,
}

/// The unified result of evaluating one chunk of text. Three buckets:
/// **expansions** (known acronyms resolved from the dictionary),
/// **extractions** (acronyms defined inline in this text), and **candidates**
/// (acronym-shaped tokens we saw but can't resolve — candidates to define).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AnalysisPayload {
    pub sentence: String,
    pub expansions: Vec<ExpansionResult>,
    pub extractions: Vec<Extraction>,
    #[serde(default)]
    pub candidates: Vec<String>,
}

impl AnalysisPayload {
    /// An empty payload that still echoes the input it was derived from.
    pub fn empty(sentence: impl Into<String>) -> Self {
        Self {
            sentence: sentence.into(),
            expansions: Vec::new(),
            extractions: Vec::new(),
            candidates: Vec::new(),
        }
    }

    /// True when nothing was expanded, extracted, or seen as a candidate.
    pub fn is_empty(&self) -> bool {
        self.expansions.is_empty() && self.extractions.is_empty() && self.candidates.is_empty()
    }
}
