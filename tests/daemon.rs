//! Daemon lifecycle: start a real background Leader, prove a Follower is served
//! by it, stop it, and confirm the janitor reaps an idle daemon.
//!
//! These spin up actual processes, so each test uses an isolated socket and
//! always tears the daemon down. They're serialized within the file by virtue
//! of distinct sockets; cleanup is best-effort on every exit path.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_ae")
}

fn scratch_socket(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("ae-daemon-{}-{label}.sock", std::process::id()));
    cleanup(&p);
    p
}

fn cleanup(socket: &Path) {
    for ext in ["sock", "db", "db-wal", "db-shm", "lock"] {
        let _ = std::fs::remove_file(socket.with_extension(ext));
    }
}

/// Run an `ae` subcommand to completion, returning (success, stdout).
fn run(socket: &Path, args: &[&str], idle_secs: &str) -> (bool, String) {
    let out = Command::new(bin())
        .arg("--socket")
        .arg(socket)
        .arg("--db")
        .arg(socket.with_extension("db"))
        .args(args)
        .env("AE_IDLE_SECS", idle_secs)
        .env("AE_CONSOLIDATE_SECS", "-1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

/// Analyze `text` as a single blob (positional arg) — the Follower path that
/// proxies to a running daemon. (Piped stdin would stream in-process instead.)
fn query(socket: &Path, text: &str) -> String {
    let out = Command::new(bin())
        .arg("--socket")
        .arg(socket)
        .arg("--db")
        .arg(socket.with_extension("db"))
        .args(["-j", text])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn connectable(socket: &Path) -> bool {
    std::os::unix::net::UnixStream::connect(socket).is_ok()
}

#[test]
fn daemon_starts_serves_a_follower_and_stops() {
    let sock = scratch_socket("lifecycle");

    let (ok, msg) = run(&sock, &["--daemon"], "30");
    assert!(ok, "daemon failed to start: {msg}");
    assert!(connectable(&sock), "socket not accepting connections");

    // A follower query is served by the daemon and returns valid JSON.
    let body = query(&sock, "Check the OKR board.");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["expansions"][0]["acronym"], "OKR");

    // Starting again is a no-op while one is running.
    let (ok2, msg2) = run(&sock, &["--daemon"], "30");
    assert!(ok2 && msg2.contains("already running"), "{msg2}");

    let (ok3, msg3) = run(&sock, &["--stop"], "30");
    assert!(ok3 && msg3.contains("stopped"), "{msg3}");

    // Give the process a moment to drop the socket.
    wait_until(Duration::from_secs(2), || !connectable(&sock));
    assert!(!connectable(&sock), "daemon still up after stop");

    cleanup(&sock);
}

#[test]
fn daemon_flag_with_input_warms_and_serves() {
    let sock = scratch_socket("dwork");
    assert!(!connectable(&sock), "no daemon should be running yet");

    // `ae -d "text"` starts the daemon AND analyzes — printing the analysis, not
    // the daemon status — and leaves the daemon warm.
    let (ok, body) = run(&sock, &["-d", "-j", "Check the OKR board."], "30");
    assert!(ok, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["expansions"][0]["acronym"], "OKR");
    assert!(
        !body.contains("\"status\""),
        "printed daemon status, not analysis: {body}"
    );
    assert!(connectable(&sock), "daemon should be left running warm");

    // The warm daemon serves a subsequent plain query.
    let v2: serde_json::Value = serde_json::from_str(&query(&sock, "Another OKR.")).unwrap();
    assert_eq!(v2["expansions"][0]["acronym"], "OKR");

    let (stopped, _) = run(&sock, &["--stop"], "30");
    assert!(stopped);
    wait_until(Duration::from_secs(2), || !connectable(&sock));
    cleanup(&sock);
}

#[test]
fn idle_daemon_reaps_itself() {
    let sock = scratch_socket("janitor");

    let (ok, _) = run(&sock, &["--daemon"], "1"); // 1-second idle timeout
    assert!(ok);
    assert!(connectable(&sock));

    // Leave it strictly alone past the timeout — note any connection (even a
    // probe) counts as activity and re-arms the janitor, so we must not poll.
    std::thread::sleep(Duration::from_secs(3));
    assert!(!connectable(&sock), "idle daemon was not reaped");

    cleanup(&sock);
}

#[test]
fn daemon_steps_down_when_its_binary_is_replaced() {
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let exe = dir.join(format!("ae-copy-{pid}"));
    let sock = dir.join(format!("ae-replace-{pid}.sock"));
    let db = dir.join(format!("ae-replace-{pid}.db"));
    let rm_all = || {
        for p in [&exe, &sock, &db, &sock.with_extension("lock")] {
            let _ = std::fs::remove_file(p);
        }
    };
    rm_all();

    // Run the daemon from a copy of the test binary so we can replace it on
    // disk. A high idle timeout means only a binary swap can reap it.
    std::fs::copy(bin(), &exe).unwrap();
    let out = Command::new(&exe)
        .arg("--daemon")
        .arg("--socket")
        .arg(&sock)
        .arg("--db")
        .arg(&db)
        .env("AE_IDLE_SECS", "3600")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
    assert!(
        out.status.success() && connectable(&sock),
        "daemon failed to start: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Replace the executable the way an upgrade does: write a fresh file and
    // atomically rename over it (new inode/mtime; rename sidesteps ETXTBSY on
    // the running binary). The janitor should notice and step down.
    let next = dir.join(format!("ae-copy-next-{pid}"));
    std::fs::copy(bin(), &next).unwrap();
    std::fs::rename(&next, &exe).unwrap();

    let stepped_down = wait_until(Duration::from_secs(3), || !connectable(&sock));
    // Best-effort: stop it if the assertion is about to fail, so a stuck daemon
    // doesn't linger for the full idle hour.
    if !stepped_down {
        let _ = Command::new(&exe)
            .arg("--stop")
            .arg("--socket")
            .arg(&sock)
            .output();
    }
    assert!(
        stepped_down,
        "daemon did not step down after its binary was replaced"
    );
    rm_all();
}

#[test]
fn status_reports_running_state_and_details() {
    let sock = scratch_socket("status");

    // No daemon: --status exits non-zero and reports not running.
    let (up0, body0) = run(&sock, &["--status", "-j"], "30");
    assert!(!up0, "status should exit non-zero with no daemon: {body0}");
    let v0: serde_json::Value = serde_json::from_str(&body0).unwrap();
    assert_eq!(v0["running"], false);

    // Start one, then --status exits zero and surfaces version + embedder + pid.
    assert!(run(&sock, &["--daemon"], "30").0);
    let (up, body) = run(&sock, &["--status", "-j"], "30");
    assert!(up, "status should exit zero while a daemon is up: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["running"], true);
    assert!(v["version"].is_string());
    assert!(v["pid"].is_number());
    assert!(v["embedder"].is_string()); // "onnx" or "hash" depending on the model

    // --status must not have started or stopped anything.
    assert!(connectable(&sock), "status probe disturbed the daemon");

    // -q is a silent health check: no output, exit code still reflects state.
    let (up_q, body_q) = run(&sock, &["--status", "-q"], "30");
    assert!(
        up_q && body_q.is_empty(),
        "quiet status should be silent + zero: {body_q:?}"
    );

    assert!(run(&sock, &["--stop"], "30").0);
    wait_until(Duration::from_secs(2), || !connectable(&sock));
    cleanup(&sock);
}

#[test]
fn stop_without_a_daemon_is_harmless() {
    let sock = scratch_socket("nostop");
    let (ok, msg) = run(&sock, &["--stop"], "30");
    assert!(ok, "stop should succeed even with no daemon");
    assert!(msg.contains("no daemon")); // reported as a status result
    cleanup(&sock);
}

/// Poll `cond` until it holds or `budget` elapses; returns whether it held.
fn wait_until(budget: Duration, cond: impl Fn() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < budget {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    cond()
}
