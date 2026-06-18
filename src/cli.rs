//! Command-line surface: argument parsing, input resolution, role dispatch.
//!
//! stdout is reserved for data; all logging is routed to stderr so a consumer
//! piping `ae` always gets clean output.

use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};

use crate::engine::Engine;
use crate::{ipc, output};

#[derive(Parser, Debug)]
#[command(name = "ae", author, version, about = "Acronym Engine")]
pub struct Cli {
    /// Context text to scan. Optional when piping input via stdin.
    pub text: Option<String>,

    /// Start a detached background daemon (the warm Leader).
    #[arg(short, long)]
    pub daemon: bool,

    /// Stop the running background daemon.
    #[arg(long)]
    pub stop: bool,

    /// Output format.
    #[arg(short, long, value_enum, default_value_t = Format::Human)]
    pub format: Format,

    /// Unix domain socket path.
    #[arg(long, default_value = "/tmp/ae.sock")]
    pub socket: PathBuf,

    /// Emit engine telemetry to stderr.
    #[arg(short, long)]
    pub verbose: bool,

    /// Internal: run as the Leader server process (spawned by `--daemon`).
    #[arg(long = "__serve", hide = true)]
    pub serve: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Human,
    Json,
    Ndjson,
}

/// Entry point — parse, set up logging, dispatch by role, render.
pub fn run() -> ExitCode {
    let cli = Cli::parse();
    init_logging(cli.verbose);

    // Internal Leader process: serve until stopped, then exit.
    if cli.serve {
        return match ipc::serve(&cli.socket) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                log::error!("server error: {e}");
                ExitCode::FAILURE
            }
        };
    }

    if cli.stop {
        return match ipc::stop(&cli.socket) {
            Ok(true) => {
                println!("ae: daemon stopped");
                ExitCode::SUCCESS
            }
            Ok(false) => {
                eprintln!("ae: no daemon running");
                ExitCode::SUCCESS
            }
            Err(e) => fail(&format!("stop failed: {e}")),
        };
    }

    if cli.daemon {
        return match ipc::start_daemon(&cli.socket) {
            Ok(ipc::DaemonOutcome::Started) => {
                println!("ae: daemon started on {}", cli.socket.display());
                ExitCode::SUCCESS
            }
            Ok(ipc::DaemonOutcome::AlreadyRunning) => {
                println!("ae: daemon already running");
                ExitCode::SUCCESS
            }
            Err(e) => fail(&format!("could not start daemon: {e}")),
        };
    }

    let text = match determine_input(&cli) {
        Ok(t) => t,
        Err(e) => return fail(&e),
    };

    let payload = match ipc::run_follower(&cli.socket, &text) {
        Ok(p) => {
            log::debug!("served by daemon");
            p
        }
        Err(_) => {
            log::warn!("no daemon; evaluating in-process");
            match evaluate_in_process(&cli, &text) {
                Ok(p) => p,
                Err(e) => return fail(&format!("evaluation failed: {e}")),
            }
        }
    };

    let stdout = io::stdout();
    if let Err(e) = output::render(&mut stdout.lock(), &payload, cli.format) {
        return fail(&format!("render failed: {e}"));
    }
    ExitCode::SUCCESS
}

/// The self-healing fallback: open the shared persistent engine and evaluate.
fn evaluate_in_process(cli: &Cli, text: &str) -> rusqlite::Result<crate::types::AnalysisPayload> {
    let engine = Engine::open(&ipc::db_path(&cli.socket))?;
    engine.analyze(text)
}

/// Resolve the text to analyze: piped stdin wins; otherwise the positional
/// argument; otherwise an error.
pub fn determine_input(cli: &Cli) -> Result<String, String> {
    if !io::stdin().is_terminal() {
        let mut buffer = String::new();
        io::stdin()
            .read_to_string(&mut buffer)
            .map_err(|e| format!("could not read stdin: {e}"))?;
        let trimmed = buffer.trim().to_string();
        if trimmed.is_empty() {
            return Err("no input on stdin".into());
        }
        Ok(trimmed)
    } else if let Some(text) = &cli.text {
        Ok(text.clone())
    } else {
        Err("no input: pass text as an argument or pipe it via stdin".into())
    }
}

fn init_logging(verbose: bool) {
    let level = if verbose { "debug" } else { "warn" };
    let _ = env_logger::Builder::new()
        .target(env_logger::Target::Stderr)
        .parse_filters(&std::env::var("RUST_LOG").unwrap_or_else(|_| level.into()))
        .try_init();
}

fn fail(msg: &str) -> ExitCode {
    eprintln!("ae: {msg}");
    ExitCode::FAILURE
}
