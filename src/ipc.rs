//! Inter-process coordination: a file lock elects one Leader (the daemon) that
//! holds the warm [`Engine`] behind a UDS; every other caller is a Follower
//! that proxies its text over the socket. When no Leader is running, callers
//! self-heal by evaluating in-process (see [`crate::cli`]).
//!
//! Wire protocol: each message is a length-prefixed frame (`u32` LE length, then
//! that many bytes). A request frame is a JSON [`Request`]; an `Analyze` reply
//! is a JSON [`AnalysisPayload`]; control replies are a one-byte ack.

use std::fs::File;
use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fs2::FileExt;
use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::types::{AnalysisPayload, StatusPayload};

/// Default idle window before a clientless daemon shuts itself down. Overridable
/// via `AE_IDLE_SECS` (tests use a short value).
const DEFAULT_IDLE_SECS: u64 = 300;
const MAX_FRAME: u32 = 64 * 1024 * 1024;
/// Two client-side waits. The write bound catches a Leader whose socket buffer
/// has filled because nobody is draining it; note that a request smaller than
/// the buffer lands instantly whether or not anyone ever reads it, so for
/// typical input the *read* bound below is what actually detects a wedged
/// Leader. Telling "unreachable" from "thinking" faster than that would take a
/// ping with its own deadline, which is a round trip on every call to save
/// three seconds on a path that shouldn't happen.
///
/// What matters either way: a Leader that stops answering must never hang its
/// callers. `ae` runs from hooks and pipelines, where a stuck process is
/// invisible until it has piled up.
const REACH_TIMEOUT: Duration = Duration::from_secs(2);
const DEFAULT_REPLY_TIMEOUT_SECS: u64 = 5;
/// The mirror image: how long the Leader waits on a client mid-request. A
/// caller that connects and stalls would otherwise pin a serving thread, and an
/// in-flight request holds the janitor's idle timer open forever.
const SERVER_TIMEOUT: Duration = Duration::from_secs(30);
/// How long graceful shutdown waits for in-flight requests before exiting
/// anyway, so a hung client can't keep the daemon alive forever.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "op", rename_all = "lowercase")]
enum Request {
    Analyze {
        text: String,
        /// Expand only — don't extract or persist new acronyms.
        #[serde(default)]
        read_only: bool,
    },
    Stop,
    Ping,
    Status,
}

/// Result of asking the OS to start a daemon.
#[derive(Debug, PartialEq)]
pub enum DaemonOutcome {
    Started,
    AlreadyRunning,
}

/// The lock file guarding single-Leader election, derived from the socket path.
pub fn lock_path(socket: &Path) -> PathBuf {
    socket.with_extension("lock")
}

fn idle_timeout() -> Duration {
    let secs = std::env::var("AE_IDLE_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_IDLE_SECS);
    // Floored at 1: zero would reap the daemon on the janitor's first tick, and
    // a zero socket timeout is an error in std — either way every call would
    // silently fall back in-process.
    Duration::from_secs(secs.max(1))
}

fn reply_timeout() -> Duration {
    let secs = std::env::var("AE_CLIENT_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REPLY_TIMEOUT_SECS);
    Duration::from_secs(secs.max(1))
}

/// Connect to a Leader with both directions bounded, so every client call either
/// completes or fails — never blocks indefinitely. The connect itself needs no
/// bound: with nothing listening it fails at once, and a Leader binds only once
/// it can answer.
fn connect(socket: &Path) -> io::Result<UnixStream> {
    let stream = UnixStream::connect(socket)?;
    stream.set_write_timeout(Some(REACH_TIMEOUT))?;
    stream.set_read_timeout(Some(reply_timeout()))?;
    Ok(stream)
}

// ---- framing -------------------------------------------------------------

fn write_frame(w: &mut impl Write, bytes: &[u8]) -> io::Result<()> {
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(bytes)?;
    w.flush()
}

fn read_frame(r: &mut impl Read) -> io::Result<Vec<u8>> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let len = u32::from_le_bytes(len);
    if len > MAX_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "frame too large",
        ));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

// ---- follower / client ---------------------------------------------------

/// Proxy `text` to a running Leader and return its analysis. `read_only`
/// requests expansion without learning. Returns `Err` when no Leader is
/// reachable, which the caller treats as "fall back in-process".
pub fn run_follower(socket: &Path, text: &str, read_only: bool) -> io::Result<AnalysisPayload> {
    let mut stream = connect(socket)?;
    let req = serde_json::to_vec(&Request::Analyze {
        text: text.to_string(),
        read_only,
    })?;
    write_frame(&mut stream, &req)?;
    let resp = read_frame(&mut stream)?;
    let payload = serde_json::from_slice(&resp)?;
    Ok(payload)
}

/// Ask a running Leader to shut down. `Ok(true)` if one was reached and told to
/// stop; `Ok(false)` if none was running.
pub fn stop(socket: &Path) -> io::Result<bool> {
    match connect(socket) {
        Ok(mut stream) => {
            write_frame(&mut stream, &serde_json::to_vec(&Request::Stop).unwrap())?;
            let _ = read_frame(&mut stream); // best-effort ack
            Ok(true)
        }
        Err(_) => Ok(false),
    }
}

/// Query a running Leader's status. `Ok(Some(_))` if one answered; `Ok(None)` if
/// none is running (unreachable socket). Read-only — never starts a daemon.
pub fn status(socket: &Path) -> io::Result<Option<StatusPayload>> {
    let Ok(mut stream) = connect(socket) else {
        return Ok(None);
    };
    write_frame(&mut stream, &serde_json::to_vec(&Request::Status).unwrap())?;
    let resp = read_frame(&mut stream)?;
    Ok(Some(serde_json::from_slice(&resp)?))
}

/// Spawn a detached daemon process for `socket`, waiting until it accepts
/// connections. A no-op (`AlreadyRunning`) if one is already up. `db` and
/// `model` are forwarded so the daemon uses the same dictionary and embedder.
pub fn start_daemon(socket: &Path, db: &Path, model: Option<&str>) -> io::Result<DaemonOutcome> {
    if UnixStream::connect(socket).is_ok() {
        return Ok(DaemonOutcome::AlreadyRunning);
    }
    let exe = std::env::current_exe()?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--__serve")
        .arg("--socket")
        .arg(socket)
        .arg("--db")
        .arg(db);
    if let Some(model) = model {
        cmd.arg("--model").arg(model);
    }
    // Its own process group: the daemon outlives the (often short-lived, often
    // signalled) caller that happened to start it.
    let mut child = cmd
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .process_group(0)
        .spawn()?;

    // Wait (up to ~3s) for the child to bind the socket.
    for _ in 0..30 {
        if UnixStream::connect(socket).is_ok() {
            return Ok(DaemonOutcome::Started);
        }
        // A child that lost the lock election exits immediately. Reap it here —
        // otherwise it lingers as a zombie for as long as we live — and stop
        // waiting for a socket it was never going to bind.
        if matches!(child.try_wait(), Ok(Some(_))) {
            return Err(io::Error::other("daemon exited during startup"));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        "daemon did not come up",
    ))
}

// ---- leader / server -----------------------------------------------------

/// Run the Leader: take the exclusive lock, bind the socket, and serve until
/// told to stop or the janitor times out. Returns early (without error) if the
/// lock is already held — another Leader won the election.
pub fn serve(socket: &Path, db: &Path, model: Option<&str>) -> io::Result<()> {
    // Snapshot our binary's identity up front — before binding the socket or the
    // slower engine load advertise readiness — so an upgrade that lands during
    // startup is still seen as a change, not baked into the baseline.
    let exe = exe_fingerprint();

    let lock_file = File::create(lock_path(socket))?;
    if lock_file.try_lock_exclusive().is_err() {
        log::info!("another leader holds the lock; exiting");
        return Ok(());
    }
    // We hold the lock for the process lifetime — keep `lock_file` alive.

    let last_activity = Arc::new(Mutex::new(Instant::now()));
    let shutdown = Arc::new(AtomicBool::new(false));

    // Start the janitor first, so it covers the engine load too. A load that
    // never returns would otherwise hold the lock forever with no watchdog
    // running — the one state no client-side timeout can rescue.
    spawn_janitor(
        socket.to_path_buf(),
        last_activity.clone(),
        shutdown.clone(),
        exe,
    );

    // Load the engine *before* binding: a connectable socket has to mean "ready
    // to answer". Binding first leaves every caller blocked on a Leader that
    // can't reply yet, and a cold model load is not instant. While we load,
    // there is no socket, so callers self-heal in-process.
    let engine = Arc::new(Mutex::new(Engine::open(db, model).map_err(to_io)?));

    // We are the sole Leader, so any socket file is stale.
    let _ = std::fs::remove_file(socket);
    let listener = UnixListener::bind(socket)?;
    log::info!("leader listening on {}", socket.display());

    let started = Instant::now();
    let active = Arc::new(AtomicUsize::new(0));

    for stream in listener.incoming() {
        // A shutdown trigger wakes this blocking accept with a throwaway
        // self-connection; observing the flag, we stop taking new work.
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                log::warn!("accept failed: {e}");
                continue;
            }
        };
        let engine = engine.clone();
        let active = active.clone();
        let last = last_activity.clone();
        let shutdown = shutdown.clone();
        let sock = socket.to_path_buf();
        let served_db = db.to_path_buf();
        std::thread::spawn(move || {
            active.fetch_add(1, Ordering::SeqCst);
            if let Err(e) =
                handle_connection(stream, &engine, &sock, &served_db, &shutdown, started)
            {
                log::warn!("connection error: {e}");
            }
            active.fetch_sub(1, Ordering::SeqCst);
            *last.lock().unwrap() = Instant::now();
        });
    }

    // Graceful shutdown: unlink the socket so new callers self-heal in-process,
    // then let in-flight requests finish before we drop the engine and exit
    // (which releases the lock). Bounded by DRAIN_TIMEOUT.
    let _ = std::fs::remove_file(socket);
    drain(&active);
    log::info!("leader stopped");
    Ok(())
}

/// Block until all in-flight connections finish, or [`DRAIN_TIMEOUT`] elapses.
fn drain(active: &AtomicUsize) {
    let deadline = Instant::now() + DRAIN_TIMEOUT;
    while active.load(Ordering::SeqCst) > 0 {
        if Instant::now() >= deadline {
            log::warn!("drain timed out with requests still in flight; exiting");
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Begin graceful shutdown: raise the flag, then wake the (blocking) accept loop
/// with a throwaway self-connection so it observes the flag promptly. Reports
/// whether the loop could be woken.
fn trigger_shutdown(shutdown: &AtomicBool, socket: &Path) -> bool {
    shutdown.store(true, Ordering::SeqCst);
    UnixStream::connect(socket).is_ok()
}

/// Shut down, one way or the other. Nothing can wake an accept loop whose
/// socket has been unlinked (or that hasn't bound one yet), and a Leader parked
/// there is immortal *and* unreachable — still holding the lock, so every
/// caller after it evaluates in-process forever. Exit instead.
fn shutdown_now(shutdown: &AtomicBool, socket: &Path) {
    if !trigger_shutdown(shutdown, socket) {
        log::warn!("cannot reach our own socket; exiting without draining");
        std::process::exit(0);
    }
}

/// A best-effort identity for the running binary. Replacing the executable on
/// disk (upgrade, reinstall, `cargo build`) changes its inode, length, or mtime,
/// so a Leader can notice it's stale and step down. `None` if it can't be read.
fn exe_fingerprint() -> Option<(u64, u64, std::time::SystemTime)> {
    use std::os::unix::fs::MetadataExt;
    let path = std::env::current_exe().ok()?;
    let md = std::fs::metadata(&path).ok()?;
    let fp = (md.ino(), md.len(), md.modified().ok()?);
    log::debug!("exe fingerprint: {path:?} -> {fp:?}");
    Some(fp)
}

fn handle_connection(
    mut stream: UnixStream,
    engine: &Mutex<Engine>,
    socket: &Path,
    served_db: &Path,
    shutdown: &AtomicBool,
    started: Instant,
) -> io::Result<()> {
    let _ = stream.set_read_timeout(Some(SERVER_TIMEOUT));
    let _ = stream.set_write_timeout(Some(SERVER_TIMEOUT));
    let req: Request = serde_json::from_slice(&read_frame(&mut stream)?)?;
    match req {
        Request::Analyze { text, read_only } => {
            let engine = engine.lock().unwrap();
            let result = if read_only {
                engine.expand_only(&text)
            } else {
                engine.analyze(&text)
            };
            let payload = result.unwrap_or_else(|e| {
                log::warn!("analysis failed: {e}");
                AnalysisPayload::empty(text)
            });
            // The warm daemon consolidates on a cadence across requests.
            if !read_only {
                let _ = engine.consolidate_if_due(
                    crate::store::PRUNE_MIN_CONFIDENCE,
                    crate::engine::prune_grace_secs(),
                );
            }
            write_frame(&mut stream, &serde_json::to_vec(&payload)?)?;
        }
        Request::Ping => write_frame(&mut stream, b"\x01")?,
        Request::Status => {
            let status = StatusPayload {
                version: env!("CARGO_PKG_VERSION").to_string(),
                pid: std::process::id(),
                uptime_secs: started.elapsed().as_secs(),
                embedder: engine.lock().unwrap().embedder_kind().to_string(),
                idle_timeout_secs: idle_timeout().as_secs(),
                db: served_db.display().to_string(),
            };
            write_frame(&mut stream, &serde_json::to_vec(&status)?)?;
        }
        Request::Stop => {
            // Ack first so the client returns promptly, then drain in the
            // background instead of hard-exiting mid-request.
            write_frame(&mut stream, b"\x01")?;
            log::info!("stop requested; draining and shutting down");
            shutdown_now(shutdown, socket);
        }
    }
    Ok(())
}

/// Watchdog: trigger a graceful shutdown when the daemon has been idle past
/// [`idle_timeout`], or when its own binary has been replaced on disk (so the
/// next call spawns a Leader running the new code). Idle is re-armed by the
/// activity timestamp every *finished* connection updates — deliberately not by
/// whether a request is in flight, or one stalled client holding a serving
/// thread would make the daemon immortal. Anything genuinely still running gets
/// [`DRAIN_TIMEOUT`] to finish.
fn spawn_janitor(
    socket: PathBuf,
    last: Arc<Mutex<Instant>>,
    shutdown: Arc<AtomicBool>,
    exe: Option<(u64, u64, std::time::SystemTime)>,
) {
    let timeout = idle_timeout();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(500));
            if shutdown.load(Ordering::SeqCst) {
                return; // shutdown already under way (e.g. via --stop)
            }
            // Stale binary: step down so an upgrade/reinstall/rebuild takes
            // effect. Only acts when we have a baseline to compare against.
            if exe.is_some() && exe_fingerprint() != exe {
                log::info!("binary changed on disk; shutting down to refresh");
                shutdown_now(&shutdown, &socket);
                return;
            }
            let elapsed = last.lock().unwrap().elapsed();
            if elapsed >= timeout {
                log::info!("idle for {:?}; shutting down", elapsed);
                shutdown_now(&shutdown, &socket);
                return;
            }
        }
    });
}

fn to_io(e: rusqlite::Error) -> io::Error {
    io::Error::other(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_path_is_derived_from_the_socket() {
        let sock = PathBuf::from("/tmp/ae-x.sock");
        assert_eq!(lock_path(&sock), PathBuf::from("/tmp/ae-x.lock"));
    }

    #[test]
    fn frames_round_trip() {
        let mut buf = Vec::new();
        write_frame(&mut buf, b"hello").unwrap();
        let got = read_frame(&mut &buf[..]).unwrap();
        assert_eq!(got, b"hello");
    }

    #[test]
    fn stop_without_a_server_reports_none() {
        let sock = std::env::temp_dir().join(format!("ae-none-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        assert!(!stop(&sock).unwrap());
    }

    #[test]
    fn follower_without_a_server_errors() {
        let sock = std::env::temp_dir().join(format!("ae-noserv-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&sock);
        assert!(run_follower(&sock, "KPI", false).is_err());
    }
}
