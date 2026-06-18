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
        &["--format", "json"],
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
    let out = run(
        &sock,
        &["--format", "ndjson"],
        Some("The OKR review is Friday."),
    );
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
    let second = run(&sock, &["--format", "json"], Some("Drain the ZQ now."));
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
    let out = run(
        &sock,
        &["--verbose", "--format", "json"],
        Some("Check the API docs."),
    );
    assert!(out.success, "stderr: {}", out.stderr);
    serde_json::from_str::<serde_json::Value>(&out.stdout).expect("stdout stayed pristine JSON");
}
