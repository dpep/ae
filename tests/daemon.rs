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

/// A stand-in Leader that binds the socket, accepts, and never answers — the
/// shape of a daemon that is still warming up or has wedged. Connections are
/// held open (closing them would hand the client an EOF, not a stall).
fn deaf_leader(socket: &Path) -> std::thread::JoinHandle<()> {
    let listener = std::os::unix::net::UnixListener::bind(socket).unwrap();
    std::thread::spawn(move || {
        let mut held = Vec::new();
        for conn in listener.incoming() {
            held.push(conn);
        }
    })
}

/// Run a child to completion under a deadline, killing it and returning `None`
/// on overrun — a hang fails the test instead of hanging the suite.
fn run_bounded(mut child: std::process::Child, limit: Duration) -> Option<std::process::Output> {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if matches!(child.try_wait(), Ok(Some(_))) {
            return Some(child.wait_with_output().unwrap());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
    None
}

/// Spawn `ae` against `socket` with a 1-second client timeout.
fn spawn(socket: &Path, args: &[&str]) -> std::process::Child {
    Command::new(bin())
        .arg("--socket")
        .arg(socket)
        .arg("--db")
        .arg(socket.with_extension("db"))
        .args(args)
        .env("AE_CLIENT_TIMEOUT_SECS", "1")
        .env("AE_CONSOLIDATE_SECS", "-1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap()
}

#[test]
fn daemon_starts_serves_a_follower_and_stops() {
    let sock = scratch_socket("lifecycle");

    let (ok, msg) = run(&sock, &["--daemon"], "30");
    assert!(ok, "daemon failed to start: {msg}");
    assert!(connectable(&sock), "socket not accepting connections");

    // A follower query is served by the daemon and returns valid JSON.
    let body = query(&sock, "Check the OKR board.");
    let v: Vec<serde_json::Value> = serde_json::from_str(&body).unwrap();
    assert!(
        v.iter()
            .any(|f| f["kind"] == "expansion" && f["acronym"] == "OKR")
    );

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
    let v: Vec<serde_json::Value> = serde_json::from_str(&body).unwrap();
    assert!(
        v.iter()
            .any(|f| f["kind"] == "expansion" && f["acronym"] == "OKR")
    );
    assert!(
        !body.contains("\"status\""),
        "printed daemon status, not analysis: {body}"
    );
    assert!(connectable(&sock), "daemon should be left running warm");

    // The warm daemon serves a subsequent plain query.
    let v2: Vec<serde_json::Value> = serde_json::from_str(&query(&sock, "Another OKR.")).unwrap();
    assert!(
        v2.iter()
            .any(|f| f["kind"] == "expansion" && f["acronym"] == "OKR")
    );

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

/// A Leader that never answers must not hang its callers: every client wait is
/// bounded, and the caller falls back to evaluating in-process.
#[test]
fn an_unresponsive_daemon_does_not_hang_a_caller() {
    let sock = scratch_socket("deaf");
    let _leader = deaf_leader(&sock);

    let out = run_bounded(
        spawn(&sock, &["-j", "Check the OKR board."]),
        Duration::from_secs(20),
    )
    .expect("caller never returned from an unresponsive daemon");
    assert!(out.status.success());
    let body = String::from_utf8_lossy(&out.stdout);
    let v: Vec<serde_json::Value> = serde_json::from_str(&body).unwrap();
    assert!(
        v.iter()
            .any(|f| f["kind"] == "expansion" && f["acronym"] == "OKR"),
        "self-healed analysis missing: {body}"
    );

    cleanup(&sock);
}

/// `-d` on a stream routes the lines through the warm daemon instead of opening
/// a private engine per invocation — the difference, for something called on
/// every command's output, between one socket round trip and one model load.
///
/// Proven by making the dictionary unopenable by anyone but the daemon that
/// already holds it: with `-d` the stream is still answered, without it the
/// same call can't even open the engine.
#[test]
fn a_stream_with_the_daemon_flag_goes_through_the_daemon() {
    use std::os::unix::fs::PermissionsExt;

    let sock = scratch_socket("dstream");
    let db = sock.with_extension("db");
    let (ok, msg) = run(&sock, &["--daemon"], "30");
    assert!(ok, "daemon failed to start: {msg}");

    let input = sock.with_extension("txt");
    std::fs::write(&input, "Check the OKR board.\n").unwrap();
    let chmod = |mode| {
        std::fs::set_permissions(&db, std::fs::Permissions::from_mode(mode)).unwrap();
    };
    let stream = |args: &[&str]| {
        run_bounded(spawn(&sock, args), Duration::from_secs(20)).expect("stream never returned")
    };
    let file = input.to_str().unwrap();

    chmod(0o000);
    let served = stream(&["-d", "-j", "--file", file]);
    let alone = stream(&["-j", "--file", file]);
    chmod(0o600);

    assert!(
        !alone.status.success(),
        "in-process stream opened a dictionary it has no access to"
    );
    assert!(served.status.success());
    let body = String::from_utf8_lossy(&served.stdout);
    let v: Vec<serde_json::Value> = serde_json::from_str(&body).unwrap();
    assert!(
        v.iter()
            .any(|f| f["kind"] == "expansion" && f["acronym"] == "OKR"),
        "daemon-served analysis missing: {body}"
    );

    let (stopped, _) = run(&sock, &["--stop"], "30");
    assert!(stopped);
    wait_until(Duration::from_secs(2), || !connectable(&sock));
    let _ = std::fs::remove_file(&input);
    cleanup(&sock);
}

/// A live Leader whose socket has gone missing still holds the lock, so a
/// caller's spawned daemon loses the election and exits at once. The caller has
/// to notice that and self-heal, not poll out its whole startup window waiting
/// for a socket that will never appear.
#[test]
fn a_daemon_that_cannot_win_the_lock_does_not_stall_its_caller() {
    let sock = scratch_socket("doomed");
    let (ok, msg) = run(&sock, &["--daemon"], "5");
    assert!(ok, "daemon failed to start: {msg}");
    std::fs::remove_file(&sock).unwrap();

    let out = run_bounded(
        spawn(&sock, &["-d", "-j", "Check the OKR board."]),
        Duration::from_secs(3),
    )
    .expect("caller waited out the daemon startup window");
    assert!(out.status.success());

    wait_until(Duration::from_secs(10), || !connectable(&sock));
    cleanup(&sock);
}

/// A client that connects and then stalls holds a serving thread for as long as
/// the Leader will wait on it. The idle clock has to keep running anyway —
/// gating it on "nothing in flight" is what makes a daemon immortal.
#[test]
fn a_stalled_client_cannot_keep_the_daemon_alive() {
    let sock = scratch_socket("stalled");
    let (ok, msg) = run(&sock, &["--daemon"], "2");
    assert!(ok, "daemon failed to start: {msg}");

    let _stalled = std::os::unix::net::UnixStream::connect(&sock).unwrap();
    // Watch the socket file, not a connection: connecting would re-arm the very
    // idle clock under test.
    assert!(
        wait_until(Duration::from_secs(10), || !sock.exists()),
        "daemon stayed up while a stalled client held a serving thread"
    );

    cleanup(&sock);
}
