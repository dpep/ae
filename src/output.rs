//! Rendering an [`AnalysisPayload`] to stdout in the requested format.
//!
//! stdout carries only data — these writers never log. Logs go to stderr via
//! `env_logger` (see [`crate::cli`]).

use std::io::Write;
use std::path::Path;

use serde_json::json;

use crate::cli::Format;
use crate::types::{AnalysisPayload, Finding, StatusPayload};

/// Render one analysis to `out` in `format`.
pub fn render(
    out: &mut impl Write,
    payload: &AnalysisPayload,
    format: Format,
) -> std::io::Result<()> {
    render_findings(out, &payload.findings(), format)
}

/// The single structured-output path. `findings` may come from one blob or an
/// aggregated stream; either way `-j` is a pretty array and `-J` is one object
/// per line, so single and stream modes emit an identical shape.
pub fn render_findings(
    out: &mut impl Write,
    findings: &[Finding],
    format: Format,
) -> std::io::Result<()> {
    match format {
        Format::Human => {
            if findings.is_empty() {
                return writeln!(out, "No acronyms found.");
            }
            for f in findings {
                write_human_finding(out, f)?;
            }
            Ok(())
        }
        Format::Json => writeln!(out, "{}", serde_json::to_string_pretty(findings).unwrap()),
        Format::Ndjson => {
            for f in findings {
                writeln!(out, "{}", serde_json::to_string(f).unwrap())?;
            }
            Ok(())
        }
    }
}

/// One finding in human (column) form: acronym, expansion, kind, confidence.
fn write_human_finding(out: &mut impl Write, f: &Finding) -> std::io::Result<()> {
    let expansion = f.expansion.as_deref().unwrap_or("(no expansion)");
    match f.confidence {
        Some(c) => writeln!(
            out,
            "{:<8} {:<40} {:<10} {:.2}",
            f.acronym, expansion, f.kind, c
        ),
        None => writeln!(out, "{:<8} {:<40} {}", f.acronym, expansion, f.kind),
    }
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

/// Render aggregated stream findings all at once. Used for the buffered
/// pretty-JSON path (which needs the whole array); delegates to the same
/// [`render_findings`] path as a single analysis.
pub fn render_lines(
    out: &mut impl Write,
    payloads: &[AnalysisPayload],
    format: Format,
) -> std::io::Result<()> {
    let findings: Vec<Finding> = payloads.iter().flat_map(|p| p.findings()).collect();
    render_findings(out, &findings, format)
}

/// Stream one analyzed line's findings, flushing so a consumer sees them
/// immediately. Human and NDJSON only — pretty JSON can't emit a partial array,
/// so callers buffer that and use [`render_lines`] at the end. Returns the
/// finding count so the caller can detect an all-empty run.
pub fn stream_line(
    out: &mut impl Write,
    payload: &AnalysisPayload,
    format: Format,
) -> std::io::Result<usize> {
    let findings = payload.findings();
    match format {
        Format::Human => {
            for f in &findings {
                write_human_finding(out, f)?;
            }
        }
        Format::Ndjson => {
            for f in &findings {
                writeln!(out, "{}", serde_json::to_string(f).unwrap())?;
            }
        }
        Format::Json => unreachable!("pretty JSON is buffered, not streamed per line"),
    }
    out.flush()?;
    Ok(findings.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ExpansionResult, Extraction, Finding, MatchCandidate};

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
    fn json_round_trips_to_the_same_findings() {
        let s = rendered(Format::Json);
        let back: Vec<Finding> = serde_json::from_str(&s).unwrap();
        assert_eq!(back, sample().findings());
    }

    #[test]
    fn ndjson_is_one_finding_per_line() {
        let s = rendered(Format::Ndjson);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 3); // expansion + extraction + candidate
        for line in lines {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(v.get("kind").is_some());
        }
    }

    #[test]
    fn json_and_ndjson_carry_the_same_objects() {
        // -j is a pretty array, -J is one-per-line — same objects, minimal diff.
        let arr: Vec<Finding> = serde_json::from_str(&rendered(Format::Json)).unwrap();
        let lines: Vec<Finding> = rendered(Format::Ndjson)
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(arr, lines);
    }

    #[test]
    fn exposed_confidence_folds_validity_into_one_score() {
        // An inline expansion (validity 0.9) with context fit 0.8 exposes a
        // single trust score 0.72 — validity is folded in, not shown separately.
        let payload = AnalysisPayload {
            sentence: "x".into(),
            expansions: vec![ExpansionResult {
                acronym: "TPS".into(),
                text_slice: "TPS".into(),
                matches: vec![MatchCandidate {
                    expansion: "Test Procedure Spec".into(),
                    validity: 0.9,
                    confidence: 0.8,
                }],
            }],
            extractions: vec![],
            candidates: vec![],
        };
        let f = &payload.findings()[0];
        assert!((f.confidence.unwrap() - 0.72).abs() < 1e-6);
        assert!(f.kind == "expansion");
    }

    #[test]
    fn stream_and_single_emit_identical_findings() {
        // The same payload rendered as a single blob and as one stream line
        // must produce byte-identical NDJSON — the core alignment guarantee.
        let mut single = Vec::new();
        render(&mut single, &sample(), Format::Ndjson).unwrap();
        let mut streamed = Vec::new();
        stream_line(&mut streamed, &sample(), Format::Ndjson).unwrap();
        assert_eq!(single, streamed);
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
