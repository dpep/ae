//! End-to-end tests that drive the built `ae` binary with an isolated socket
//! and DB, exercising the no-daemon self-healing fallback path.
//!
//! Input is fed via stdin (the pipe path): when stdin isn't a TTY — which is
//! always the case under the test harness — `ae` consumes it, so we pipe text
//! rather than passing it as an argument.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// A unique socket path per test, so the derived `.db`/`.lock` are isolated and
/// tests can run in parallel.
fn scratch_socket(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("ae-e2e-{}-{label}.sock", std::process::id()));
    for ext in ["sock", "db", "db-wal", "db-shm", "lock"] {
        let _ = std::fs::remove_file(p.with_extension(ext));
    }
    p
}

struct Output {
    success: bool,
    stdout: String,
    stderr: String,
}

/// Run `ae` with `args`, feeding `stdin` (or an empty closed stdin if `None`).
fn run(socket: &std::path::Path, args: &[&str], stdin: Option<&str>) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_ae"));
    cmd.arg("--socket")
        .arg(socket)
        .arg("--db")
        .arg(socket.with_extension("db"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().unwrap();
    if let Some(text) = stdin {
        child
            .stdin
            .take()
            .unwrap()
            .write_all(text.as_bytes())
            .unwrap();
    } // dropping the handle closes stdin (EOF) either way
    let out = child.wait_with_output().unwrap();
    Output {
        success: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

#[test]
fn pipes_text_and_expands_a_known_acronym() {
    let sock = scratch_socket("human");
    let out = run(&sock, &[], Some("Check the OKR board today."));
    assert!(out.success, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Objectives and Key Results"));
    assert!(out.stdout.contains("expansion"));
}

#[test]
fn json_output_parses_and_reports_learning() {
    let sock = scratch_socket("json");
    let out = run(
        &sock,
        &["-j"],
        Some("Our KPI (Key Performance Indicator) is up."),
    );
    assert!(out.success, "stderr: {}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert!(v["sentence"].is_string());
    let learned = v["learned_candidates"].as_array().unwrap();
    assert!(learned.iter().any(|c| c["acronym"] == "KPI"));
}

#[test]
fn ndjson_emits_one_object_per_line() {
    let sock = scratch_socket("ndjson");
    let out = run(&sock, &["-J"], Some("The OKR review is Friday."));
    assert!(out.success, "stderr: {}", out.stderr);
    let mut lines = out.stdout.lines();
    let first: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    assert_eq!(first["kind"], "expansion");
    assert_eq!(first["acronym"], "OKR");
}

#[test]
fn learning_persists_across_invocations_via_shared_db() {
    let sock = scratch_socket("persist");
    // First pass teaches ZQ.
    let first = run(&sock, &[], Some("The ZQ (Zebra Queue) is deep."));
    assert!(first.stdout.contains("learned"));
    // Second pass — same socket → same DB — now expands it.
    let second = run(&sock, &["-j"], Some("Drain the ZQ now."));
    let v: serde_json::Value = serde_json::from_str(&second.stdout).unwrap();
    let expansions = v["expansions"].as_array().unwrap();
    assert!(
        expansions
            .iter()
            .any(|e| e["acronym"] == "ZQ" && e["matches"][0]["expansion"] == "Zebra Queue"),
        "ZQ was not expanded on the second pass: {}",
        second.stdout
    );
}

#[test]
fn model_flag_is_accepted_and_degrades_gracefully() {
    let sock = scratch_socket("model");
    // A bogus model path must not break the run — it falls back to the hash
    // embedder, and expansion (trie + dictionary) is unaffected.
    let out = run(
        &sock,
        &["--model", "/no/such/model", "-j"],
        Some("Check the OKR board."),
    );
    assert!(out.success, "stderr: {}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["expansions"][0]["acronym"], "OKR");
}

#[test]
fn read_only_expands_without_learning_or_persisting() {
    let sock = scratch_socket("readonly");
    // Read-only: inline definition present, but nothing is learned...
    let first = run(
        &sock,
        &["--read-only", "-j"],
        Some("The ZQ (Zebra Queue) backs the OKR."),
    );
    let v: serde_json::Value = serde_json::from_str(&first.stdout).unwrap();
    assert!(
        v["expansions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["acronym"] == "OKR")
    );
    assert!(v["learned_candidates"].as_array().unwrap().is_empty());

    // ...and nothing was persisted: a normal later pass still doesn't know ZQ.
    let second = run(&sock, &["-j"], Some("Drain the ZQ."));
    let v2: serde_json::Value = serde_json::from_str(&second.stdout).unwrap();
    assert!(
        !v2["expansions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["acronym"] == "ZQ")
    );
}

#[test]
fn short_json_and_ndjson_flags_select_the_format() {
    let sock = scratch_socket("shortflags");
    // -j → a single pretty JSON object.
    let j = run(&sock, &["-j"], Some("Check the OKR board."));
    serde_json::from_str::<serde_json::Value>(&j.stdout).expect("-j is valid JSON");

    // -J → one compact object per line.
    let nd = run(&sock, &["-J"], Some("Check the OKR board."));
    let line = nd.stdout.lines().next().unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(line).unwrap()["kind"],
        "expansion"
    );
}

#[test]
fn commands_emit_status_json_in_machine_mode() {
    let sock = scratch_socket("status");
    // No daemon running → stop reports it as a JSON status object on stdout.
    let out = run(&sock, &["--stop", "-j"], None);
    assert!(out.success);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["status"], "not_running");
}

#[test]
fn batch_mode_aggregates_per_line_hits_with_positions() {
    let sock = scratch_socket("batch");
    let input = "first line has an OKR\nsecond mentions the API\n";
    let out = run(&sock, &["--batch", "-j"], Some(input));
    assert!(out.success, "stderr: {}", out.stderr);
    let hits: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let arr = hits.as_array().unwrap();
    assert!(
        arr.iter()
            .any(|h| h["acronym"] == "OKR" && h["line"] == 1 && h["col"].as_u64().unwrap() > 0)
    );
    assert!(arr.iter().any(|h| h["acronym"] == "API" && h["line"] == 2));
}

#[test]
fn file_flag_reads_a_file_and_implies_batch() {
    let sock = scratch_socket("file");
    let path = std::env::temp_dir().join(format!("ae-input-{}.txt", std::process::id()));
    std::fs::write(&path, "intro line\nthis row has the OKR\n").unwrap();
    // No stdin, no --batch — the file alone triggers aggregated output.
    let out = run(&sock, &["--file", path.to_str().unwrap(), "-j"], None);
    assert!(out.success, "stderr: {}", out.stderr);
    let hits: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert!(
        hits.as_array()
            .unwrap()
            .iter()
            .any(|h| h["acronym"] == "OKR" && h["line"] == 2)
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn batch_human_output_is_grep_style() {
    let sock = scratch_socket("batchhuman");
    let out = run(&sock, &["-b"], Some("line one\nthe OKR is here\n"));
    assert!(out.success, "stderr: {}", out.stderr);
    // line:col: ACR ... — the OKR is on line 2.
    assert!(
        out.stdout
            .lines()
            .any(|l| l.starts_with("2:") && l.contains("OKR"))
    );
}

#[test]
fn unknown_acronyms_are_surfaced() {
    let sock = scratch_socket("unknown");
    let out = run(&sock, &["-j"], Some("hi there MVP"));
    assert!(out.success, "stderr: {}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let unknown = v["unknown"].as_array().unwrap();
    assert!(
        unknown.iter().any(|u| u == "MVP"),
        "MVP not surfaced: {}",
        out.stdout
    );
}

#[test]
fn plain_text_reports_no_findings() {
    let sock = scratch_socket("plain");
    let out = run(&sock, &[], Some("just an ordinary lowercase sentence"));
    assert!(out.success);
    assert!(out.stdout.contains("No acronyms"));
}

#[test]
fn empty_input_is_an_error() {
    let sock = scratch_socket("empty");
    let out = run(&sock, &[], Some(""));
    assert!(!out.success);
    assert!(out.stderr.contains("no input"));
}

#[test]
fn logs_go_to_stderr_not_stdout() {
    let sock = scratch_socket("streams");
    // --verbose forces telemetry; stdout must still be parseable JSON.
    let out = run(&sock, &["--verbose", "-j"], Some("Check the API docs."));
    assert!(out.success, "stderr: {}", out.stderr);
    serde_json::from_str::<serde_json::Value>(&out.stdout).expect("stdout stayed pristine JSON");
}
