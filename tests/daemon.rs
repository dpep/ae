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

/// Pipe `text` into an `ae` invocation (the Follower path when a daemon is up).
fn query(socket: &Path, text: &str) -> String {
    use std::io::Write;
    let mut child = Command::new(bin())
        .arg("--socket")
        .arg(socket)
        .arg("--db")
        .arg(socket.with_extension("db"))
        .args(["-j"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(text.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
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
