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
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fs2::FileExt;
use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::types::AnalysisPayload;

/// Default idle window before a clientless daemon shuts itself down. Overridable
/// via `AE_IDLE_SECS` (tests use a short value).
const DEFAULT_IDLE_SECS: u64 = 300;
const MAX_FRAME: u32 = 64 * 1024 * 1024;

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
    Duration::from_secs(secs)
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
    let mut stream = UnixStream::connect(socket)?;
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
    match UnixStream::connect(socket) {
        Ok(mut stream) => {
            write_frame(&mut stream, &serde_json::to_vec(&Request::Stop).unwrap())?;
            let _ = read_frame(&mut stream); // best-effort ack
            Ok(true)
        }
        Err(_) => Ok(false),
    }
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
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;

    // Wait (up to ~3s) for the child to bind the socket.
    for _ in 0..30 {
        if UnixStream::connect(socket).is_ok() {
            return Ok(DaemonOutcome::Started);
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
    let lock_file = File::create(lock_path(socket))?;
    if lock_file.try_lock_exclusive().is_err() {
        log::info!("another leader holds the lock; exiting");
        return Ok(());
    }
    // We hold the lock for the process lifetime — keep `lock_file` alive.

    // We are the sole Leader, so any socket file is stale.
    let _ = std::fs::remove_file(socket);
    let listener = UnixListener::bind(socket)?;
    log::info!("leader listening on {}", socket.display());

    let engine = Arc::new(Mutex::new(Engine::open(db, model).map_err(to_io)?));
    let active = Arc::new(AtomicUsize::new(0));
    let last_activity = Arc::new(Mutex::new(Instant::now()));

    spawn_janitor(socket.to_path_buf(), active.clone(), last_activity.clone());

    for stream in listener.incoming() {
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
        let sock = socket.to_path_buf();
        std::thread::spawn(move || {
            active.fetch_add(1, Ordering::SeqCst);
            if let Err(e) = handle_connection(stream, &engine, &sock) {
                log::warn!("connection error: {e}");
            }
            active.fetch_sub(1, Ordering::SeqCst);
            *last.lock().unwrap() = Instant::now();
        });
    }
    Ok(())
}

fn handle_connection(
    mut stream: UnixStream,
    engine: &Mutex<Engine>,
    socket: &Path,
) -> io::Result<()> {
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
            write_frame(&mut stream, &serde_json::to_vec(&payload)?)?;
        }
        Request::Ping => write_frame(&mut stream, b"\x01")?,
        Request::Stop => {
            write_frame(&mut stream, b"\x01")?;
            log::info!("stop requested; shutting down");
            let _ = std::fs::remove_file(socket);
            std::process::exit(0);
        }
    }
    Ok(())
}

/// Watchdog: once no connection has been active for [`idle_timeout`], remove the
/// socket and exit. Re-armed by the activity timestamp every connection updates.
fn spawn_janitor(socket: PathBuf, active: Arc<AtomicUsize>, last: Arc<Mutex<Instant>>) {
    let timeout = idle_timeout();
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_millis(500));
            let idle = active.load(Ordering::SeqCst) == 0;
            let elapsed = last.lock().unwrap().elapsed();
            if idle && elapsed >= timeout {
                log::info!("idle for {:?}; shutting down", elapsed);
                let _ = std::fs::remove_file(&socket);
                std::process::exit(0);
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
