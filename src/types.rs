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

/// A newly discovered acronym/definition pair extracted from inline structure
/// (e.g. `KPI (Key Performance Indicator)`), not yet necessarily in the
/// dictionary.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct LearnedCandidate {
    pub acronym: String,
    pub extracted_definition: String,
    /// Which rule matched — e.g. `"alpha"` or `"beta"`.
    pub pattern_type: String,
    pub confidence: f32,
}

/// The unified result of evaluating one chunk of text.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AnalysisPayload {
    pub sentence: String,
    pub expansions: Vec<ExpansionResult>,
    pub learned_candidates: Vec<LearnedCandidate>,
}

impl AnalysisPayload {
    /// An empty payload that still echoes the input it was derived from.
    pub fn empty(sentence: impl Into<String>) -> Self {
        Self {
            sentence: sentence.into(),
            expansions: Vec::new(),
            learned_candidates: Vec::new(),
        }
    }

    /// True when nothing was expanded and nothing was learned.
    pub fn is_empty(&self) -> bool {
        self.expansions.is_empty() && self.learned_candidates.is_empty()
    }
}
