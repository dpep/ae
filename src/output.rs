//! Rendering an [`AnalysisPayload`] to stdout in the requested format.
//!
//! stdout carries only data — these writers never log. Logs go to stderr via
//! `env_logger` (see [`crate::cli`]).

use std::io::Write;
use std::path::Path;

use serde_json::json;

use crate::cli::Format;
use crate::types::{AnalysisPayload, StatusPayload};

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
            // validity = is this a real expansion; confidence = fit for this context.
            writeln!(
                out,
                "{:<8} {:<40} expansion  v{:.2} c{:.2}",
                r.acronym, m.expansion, m.validity, m.confidence
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

/// Render `(acronym, expansion, source)` dictionary entries (list/show). The
/// `source` (user/inline) is shown so the verified status is visible.
pub fn render_entries(
    out: &mut impl Write,
    entries: &[(String, String, String)],
    format: Format,
) -> std::io::Result<()> {
    match format {
        Format::Human => {
            if entries.is_empty() {
                return writeln!(out, "No acronyms.");
            }
            for (acronym, expansion, source) in entries {
                writeln!(out, "{acronym:<8} {expansion:<40} {source}")?;
            }
        }
        Format::Json => {
            let rows: Vec<_> = entries
                .iter()
                .map(|(a, e, s)| {
                    json!({ "acronym": a, "expansion": e, "source": s, "verified": s == "user" })
                })
                .collect();
            writeln!(out, "{}", serde_json::to_string_pretty(&rows).unwrap())?;
        }
        Format::Ndjson => {
            for (acronym, expansion, source) in entries {
                writeln!(
                    out,
                    "{}",
                    json!({ "acronym": acronym, "expansion": expansion, "source": source, "verified": source == "user" })
                )?;
            }
        }
    }
    Ok(())
}

/// Render candidate acronyms with `(count, source, on-watch-list)` — the
/// provenance and watch state, for observability.
pub fn render_candidates(
    out: &mut impl Write,
    candidates: &[(String, i64, String, bool)],
    format: Format,
) -> std::io::Result<()> {
    match format {
        Format::Human => {
            if candidates.is_empty() {
                return writeln!(out, "No candidates.");
            }
            for (acronym, count, source, watching) in candidates {
                let watch = if *watching { "watching" } else { "" };
                writeln!(out, "{acronym:<8} {count:>4}  {source:<9} {watch}")?;
            }
        }
        Format::Json => {
            let rows: Vec<_> = candidates
                .iter()
                .map(|(a, n, s, w)| json!({ "acronym": a, "count": n, "source": s, "watching": w }))
                .collect();
            writeln!(out, "{}", serde_json::to_string_pretty(&rows).unwrap())?;
        }
        Format::Ndjson => {
            for (acronym, count, source, watching) in candidates {
                writeln!(
                    out,
                    "{}",
                    json!({ "acronym": acronym, "count": count, "source": source, "watching": watching })
                )?;
            }
        }
    }
    Ok(())
}

/// Render speculative expansions: `(acronym, expansion, count, confidence)`.
pub fn render_suggestions(
    out: &mut impl Write,
    rows: &[(String, String, i64, f32)],
    format: Format,
) -> std::io::Result<()> {
    match format {
        Format::Human => {
            if rows.is_empty() {
                return writeln!(out, "No suggestions yet.");
            }
            for (acronym, expansion, count, confidence) in rows {
                writeln!(
                    out,
                    "{acronym:<8} {expansion:<40} {confidence:.2} ({count})"
                )?;
            }
        }
        Format::Json => {
            let items: Vec<_> = rows
                .iter()
                .map(|(a, e, n, c)| json!({ "acronym": a, "expansion": e, "count": n, "confidence": c }))
                .collect();
            writeln!(out, "{}", serde_json::to_string_pretty(&items).unwrap())?;
        }
        Format::Ndjson => {
            for (a, e, n, c) in rows {
                writeln!(
                    out,
                    "{}",
                    json!({ "acronym": a, "expansion": e, "count": n, "confidence": c })
                )?;
            }
        }
    }
    Ok(())
}

/// Render daemon status. `report` is `Some` when a daemon answered, `None` when
/// none is running; `socket`/`db` are the paths the CLI resolved (what was
/// checked). Honors the output format like every other command.
pub fn render_status(
    out: &mut impl Write,
    report: Option<&StatusPayload>,
    socket: &Path,
    db: &Path,
    format: Format,
) -> std::io::Result<()> {
    match format {
        Format::Human => match report {
            Some(s) => {
                writeln!(
                    out,
                    "ae: daemon running (pid {}, up {})",
                    s.pid,
                    fmt_uptime(s.uptime_secs)
                )?;
                writeln!(out, "  version    {}", s.version)?;
                writeln!(out, "  embedder   {}", s.embedder)?;
                writeln!(out, "  idle       {}s", s.idle_timeout_secs)?;
                writeln!(out, "  socket     {}", socket.display())?;
                writeln!(out, "  db         {}", db.display())?;
            }
            None => {
                writeln!(out, "ae: no daemon running")?;
                writeln!(out, "  socket     {}", socket.display())?;
                writeln!(out, "  db         {}", db.display())?;
            }
        },
        Format::Json => writeln!(
            out,
            "{}",
            serde_json::to_string_pretty(&status_json(report, socket, db)).unwrap()
        )?,
        Format::Ndjson => writeln!(out, "{}", status_json(report, socket, db))?,
    }
    Ok(())
}

fn status_json(report: Option<&StatusPayload>, socket: &Path, db: &Path) -> serde_json::Value {
    match report {
        Some(s) => json!({
            "running": true,
            "version": s.version,
            "pid": s.pid,
            "uptime_secs": s.uptime_secs,
            "embedder": s.embedder,
            "idle_timeout_secs": s.idle_timeout_secs,
            "socket": socket.display().to_string(),
            "db": db.display().to_string(),
        }),
        None => json!({
            "running": false,
            "socket": socket.display().to_string(),
            "db": db.display().to_string(),
        }),
    }
}

/// Human-friendly uptime: `45s`, `3m12s`, `2h05m`.
fn fmt_uptime(secs: u64) -> String {
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}h{m:02}m")
    } else if m > 0 {
        format!("{m}m{s:02}s")
    } else {
        format!("{s}s")
    }
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
                "validity": m.validity,
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
                    validity: 1.0,
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

    fn status_sample() -> StatusPayload {
        StatusPayload {
            version: "9.9.9".into(),
            pid: 4242,
            uptime_secs: 75,
            embedder: "onnx".into(),
            idle_timeout_secs: 300,
        }
    }

    fn status_rendered(report: Option<&StatusPayload>, format: Format) -> String {
        let mut buf = Vec::new();
        render_status(
            &mut buf,
            report,
            Path::new("/tmp/ae.sock"),
            Path::new("/tmp/ae.db"),
            format,
        )
        .unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn status_human_surfaces_version_and_embedder() {
        let s = status_sample();
        let out = status_rendered(Some(&s), Format::Human);
        assert!(out.contains("running") && out.contains("9.9.9") && out.contains("onnx"));
    }

    #[test]
    fn status_json_reflects_running_state() {
        let s = status_sample();
        let up: serde_json::Value =
            serde_json::from_str(&status_rendered(Some(&s), Format::Json)).unwrap();
        assert_eq!(up["running"], true);
        assert_eq!(up["embedder"], "onnx");
        assert_eq!(up["version"], "9.9.9");

        let down: serde_json::Value =
            serde_json::from_str(&status_rendered(None, Format::Json)).unwrap();
        assert_eq!(down["running"], false);
        assert_eq!(down["socket"], "/tmp/ae.sock");
    }
}
