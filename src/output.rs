//! Rendering an [`AnalysisPayload`] to stdout in the requested format.
//!
//! stdout carries only data — these writers never log. Logs go to stderr via
//! `env_logger` (see [`crate::cli`]).

use std::io::Write;

use serde_json::json;

use crate::cli::Format;
use crate::types::AnalysisPayload;

/// Render `payload` to `out` in `format`.
pub fn render(
    out: &mut impl Write,
    payload: &AnalysisPayload,
    format: Format,
) -> std::io::Result<()> {
    match format {
        Format::Human => render_human(out, payload),
        Format::Json => writeln!(out, "{}", serde_json::to_string_pretty(payload).unwrap()),
        Format::Ndjson => render_ndjson(out, payload),
    }
}

fn render_human(out: &mut impl Write, payload: &AnalysisPayload) -> std::io::Result<()> {
    if payload.is_empty() {
        return writeln!(out, "No acronyms found.");
    }
    for r in &payload.expansions {
        for m in &r.matches {
            writeln!(
                out,
                "{:<8} {:<40} expansion {:.2}",
                r.acronym, m.expansion, m.confidence
            )?;
        }
    }
    for c in &payload.extractions {
        writeln!(
            out,
            "{:<8} {:<40} extraction {:.2}",
            c.acronym, c.extracted_definition, c.confidence
        )?;
    }
    for acronym in &payload.candidates {
        writeln!(out, "{:<8} {:<40} candidate", acronym, "(no expansion)")?;
    }
    Ok(())
}

/// Render a list of `(acronym, expansion)` dictionary entries (list/show/search).
pub fn render_entries(
    out: &mut impl Write,
    entries: &[(String, String)],
    format: Format,
) -> std::io::Result<()> {
    match format {
        Format::Human => {
            if entries.is_empty() {
                return writeln!(out, "No acronyms.");
            }
            for (acronym, expansion) in entries {
                writeln!(out, "{acronym:<8} {expansion}")?;
            }
        }
        Format::Json => {
            let rows: Vec<_> = entries
                .iter()
                .map(|(a, e)| json!({ "acronym": a, "expansion": e }))
                .collect();
            writeln!(out, "{}", serde_json::to_string_pretty(&rows).unwrap())?;
        }
        Format::Ndjson => {
            for (acronym, expansion) in entries {
                writeln!(
                    out,
                    "{}",
                    json!({ "acronym": acronym, "expansion": expansion })
                )?;
            }
        }
    }
    Ok(())
}

/// Render candidate acronyms with their occurrence counts (most-seen first).
pub fn render_candidates(
    out: &mut impl Write,
    candidates: &[(String, i64)],
    format: Format,
) -> std::io::Result<()> {
    match format {
        Format::Human => {
            if candidates.is_empty() {
                return writeln!(out, "No candidates.");
            }
            for (acronym, count) in candidates {
                writeln!(out, "{acronym:<8} {count}")?;
            }
        }
        Format::Json => {
            let rows: Vec<_> = candidates
                .iter()
                .map(|(a, n)| json!({ "acronym": a, "count": n }))
                .collect();
            writeln!(out, "{}", serde_json::to_string_pretty(&rows).unwrap())?;
        }
        Format::Ndjson => {
            for (acronym, count) in candidates {
                writeln!(out, "{}", json!({ "acronym": acronym, "count": count }))?;
            }
        }
    }
    Ok(())
}

/// One analyzed line of a batch run: its number, original text, and findings.
pub struct LineResult {
    pub line: usize,
    pub text: String,
    pub payload: AnalysisPayload,
}

/// A single position-tagged finding in batch output.
#[derive(serde::Serialize)]
struct Hit {
    kind: &'static str,
    line: usize,
    col: usize,
    acronym: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    expansion: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pattern_type: Option<String>,
}

/// Render aggregated batch findings as `line:col`-tagged hits.
pub fn render_lines(
    out: &mut impl Write,
    results: &[LineResult],
    format: Format,
) -> std::io::Result<()> {
    let hits = build_hits(results);
    match format {
        Format::Human => {
            if hits.is_empty() {
                return writeln!(out, "No acronyms found.");
            }
            for h in &hits {
                let conf = h.confidence.map(|c| format!("{c:.2}")).unwrap_or_default();
                let expansion = if h.expansion.is_empty() {
                    "(no expansion)"
                } else {
                    &h.expansion
                };
                writeln!(
                    out,
                    "{}:{}: {:<8} {:<40} {:<9} {}",
                    h.line, h.col, h.acronym, expansion, h.kind, conf
                )?;
            }
        }
        Format::Json => writeln!(out, "{}", serde_json::to_string_pretty(&hits).unwrap())?,
        Format::Ndjson => {
            for h in &hits {
                writeln!(out, "{}", serde_json::to_string(h).unwrap())?;
            }
        }
    }
    Ok(())
}

fn build_hits(results: &[LineResult]) -> Vec<Hit> {
    let mut hits = Vec::new();
    for r in results {
        for e in &r.payload.expansions {
            let col = col_of(&r.text, &e.text_slice);
            for m in &e.matches {
                hits.push(Hit {
                    kind: "expansion",
                    line: r.line,
                    col,
                    acronym: e.acronym.clone(),
                    expansion: m.expansion.clone(),
                    confidence: Some(m.confidence),
                    pattern_type: None,
                });
            }
        }
        for c in &r.payload.extractions {
            hits.push(Hit {
                kind: "extraction",
                line: r.line,
                col: col_of(&r.text, &c.acronym),
                acronym: c.acronym.clone(),
                expansion: c.extracted_definition.clone(),
                confidence: Some(c.confidence),
                pattern_type: Some(c.pattern_type.clone()),
            });
        }
        for acronym in &r.payload.candidates {
            hits.push(Hit {
                kind: "candidate",
                line: r.line,
                col: col_of(&r.text, acronym),
                acronym: acronym.clone(),
                expansion: String::new(),
                confidence: None,
                pattern_type: None,
            });
        }
    }
    hits
}

/// 1-indexed character column where `needle` first appears in `line`.
fn col_of(line: &str, needle: &str) -> usize {
    match line.find(needle) {
        Some(byte) => line[..byte].chars().count() + 1,
        None => 1,
    }
}

fn render_ndjson(out: &mut impl Write, payload: &AnalysisPayload) -> std::io::Result<()> {
    for r in &payload.expansions {
        for m in &r.matches {
            let line = json!({
                "kind": "expansion",
                "acronym": r.acronym,
                "text_slice": r.text_slice,
                "expansion": m.expansion,
                "confidence": m.confidence,
            });
            writeln!(out, "{line}")?;
        }
    }
    for c in &payload.extractions {
        let line = json!({
            "kind": "extraction",
            "acronym": c.acronym,
            "expansion": c.extracted_definition,
            "pattern_type": c.pattern_type,
            "confidence": c.confidence,
        });
        writeln!(out, "{line}")?;
    }
    for acronym in &payload.candidates {
        let line = json!({ "kind": "candidate", "acronym": acronym });
        writeln!(out, "{line}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ExpansionResult, Extraction, MatchCandidate};

    fn sample() -> AnalysisPayload {
        AnalysisPayload {
            sentence: "KPI (Key Performance Indicator)".into(),
            expansions: vec![ExpansionResult {
                acronym: "KPI".into(),
                text_slice: "KPI".into(),
                matches: vec![MatchCandidate {
                    expansion: "Key Performance Indicator".into(),
                    confidence: 0.8,
                }],
            }],
            extractions: vec![Extraction {
                acronym: "KPI".into(),
                extracted_definition: "Key Performance Indicator".into(),
                pattern_type: "alpha".into(),
                confidence: 0.95,
            }],
            candidates: vec!["MVP".into()],
        }
    }

    fn rendered(format: Format) -> String {
        let mut buf = Vec::new();
        render(&mut buf, &sample(), format).unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn human_mentions_each_category() {
        let s = rendered(Format::Human);
        assert!(s.contains("expansion"));
        assert!(s.contains("extraction"));
        assert!(s.contains("Key Performance Indicator"));
        assert!(s.contains("candidate") && s.contains("MVP"));
    }

    #[test]
    fn json_round_trips_to_the_same_payload() {
        let s = rendered(Format::Json);
        let back: AnalysisPayload = serde_json::from_str(&s).unwrap();
        assert_eq!(back, sample());
    }

    #[test]
    fn ndjson_is_one_object_per_line() {
        let s = rendered(Format::Ndjson);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 3); // expansion + extraction + candidate
        for line in lines {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(v.get("kind").is_some());
        }
    }

    #[test]
    fn human_empty_payload_says_so() {
        let mut buf = Vec::new();
        render(&mut buf, &AnalysisPayload::empty("x"), Format::Human).unwrap();
        assert!(String::from_utf8(buf).unwrap().contains("No acronyms"));
    }
}
