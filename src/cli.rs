//! Command-line surface: argument parsing, input resolution, role dispatch.
//!
//! stdout is reserved for data; all logging is routed to stderr so a consumer
//! piping `ae` always gets clean output.

use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{CommandFactory, Parser};

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

    /// JSON output: a pretty object (analysis) or `{"status": …}` (commands).
    #[arg(short = 'j', long, conflicts_with = "ndjson")]
    pub json: bool,

    /// NDJSON output: one compact object per finding/line.
    #[arg(short = 'J', long)]
    pub ndjson: bool,

    /// Expand known acronyms only — don't extract or learn new ones (no writes).
    #[arg(short, long)]
    pub read_only: bool,

    /// Batch mode: analyze input line by line (e.g. `cat file | ae -b`) and
    /// aggregate the findings, each tagged with its `line:col` position.
    #[arg(short, long)]
    pub batch: bool,

    /// Read input from this file, analyzed line by line (implies `--batch`).
    #[arg(short, long)]
    pub file: Option<PathBuf>,

    /// Unix domain socket path.
    #[arg(long, default_value = "/tmp/ae.sock")]
    pub socket: PathBuf,

    /// Embedding model: a path (directory or `.onnx` file) or a name resolved
    /// against the model search dirs. Defaults to the bundled/cached model.
    #[arg(short, long)]
    pub model: Option<String>,

    /// Emit engine telemetry to stderr.
    #[arg(short, long)]
    pub verbose: bool,

    /// Internal: run as the Leader server process (spawned by `--daemon`).
    #[arg(long = "__serve", hide = true)]
    pub serve: bool,
}

impl Cli {
    /// The effective output format. Default is human; `-J/--ndjson` wins over
    /// `-j/--json`.
    pub fn format(&self) -> Format {
        if self.ndjson {
            Format::Ndjson
        } else if self.json {
            Format::Json
        } else {
            Format::Human
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Human,
    Json,
    Ndjson,
}

/// Entry point — parse, set up logging, dispatch by role, render.
pub fn run() -> ExitCode {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    let fmt = cli.format();

    // Internal Leader process: serve until stopped, then exit.
    if cli.serve {
        return match ipc::serve(&cli.socket, cli.model.as_deref()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                log::error!("server error: {e}");
                ExitCode::FAILURE
            }
        };
    }

    if cli.stop {
        return match ipc::stop(&cli.socket) {
            Ok(true) => status(fmt, "stopped", "daemon stopped"),
            Ok(false) => status(fmt, "not_running", "no daemon running"),
            Err(e) => fail(fmt, &format!("stop failed: {e}")),
        };
    }

    if cli.daemon {
        return match ipc::start_daemon(&cli.socket, cli.model.as_deref()) {
            Ok(ipc::DaemonOutcome::Started) => status(fmt, "started", "daemon started"),
            Ok(ipc::DaemonOutcome::AlreadyRunning) => {
                status(fmt, "already_running", "daemon already running")
            }
            Err(e) => fail(fmt, &format!("could not start daemon: {e}")),
        };
    }

    // A file or explicit batch flag runs the aggregated line-by-line path.
    if cli.batch || cli.file.is_some() {
        return run_batch(&cli, fmt);
    }

    // Bare invocation with nothing to analyze: show help instead of an error.
    if cli.text.is_none() && io::stdin().is_terminal() {
        let _ = Cli::command().print_help();
        println!();
        return ExitCode::SUCCESS;
    }

    let text = match determine_input(&cli) {
        Ok(t) => t,
        Err(e) => return fail(fmt, &e),
    };

    let payload = match ipc::run_follower(&cli.socket, &text, cli.read_only) {
        Ok(p) => {
            log::debug!("served by daemon");
            p
        }
        Err(_) => {
            log::debug!("no daemon; evaluating in-process");
            match evaluate_in_process(&cli, &text) {
                Ok(p) => p,
                Err(e) => return fail(fmt, &format!("evaluation failed: {e}")),
            }
        }
    };

    let stdout = io::stdout();
    if let Err(e) = output::render(&mut stdout.lock(), &payload, fmt) {
        return fail(fmt, &format!("render failed: {e}"));
    }
    ExitCode::SUCCESS
}

/// The self-healing fallback: open the shared persistent engine and evaluate.
/// Honors `--read-only` (expand without learning).
fn evaluate_in_process(cli: &Cli, text: &str) -> rusqlite::Result<crate::types::AnalysisPayload> {
    let engine = Engine::open(&ipc::db_path(&cli.socket), cli.model.as_deref())?;
    if cli.read_only {
        engine.expand_only(text)
    } else {
        engine.analyze(text)
    }
}

/// Batch mode: analyze each input line with one warm in-process engine and emit
/// aggregated, position-tagged hits. Runs in-process (not via the daemon) since
/// it's one bulk pass over many lines.
fn run_batch(cli: &Cli, fmt: Format) -> ExitCode {
    let raw = match &cli.file {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => return fail(fmt, &format!("could not read {}: {e}", path.display())),
        },
        None => match read_raw_input(cli) {
            Ok(r) => r,
            Err(e) => return fail(fmt, &e),
        },
    };
    let engine = match Engine::open(&ipc::db_path(&cli.socket), cli.model.as_deref()) {
        Ok(e) => e,
        Err(e) => return fail(fmt, &format!("could not open engine: {e}")),
    };

    let mut results = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let payload = if cli.read_only {
            engine.expand_only(line)
        } else {
            engine.analyze(line)
        };
        match payload {
            Ok(p) if !p.is_empty() => results.push(output::LineResult {
                line: i + 1,
                text: line.to_string(),
                payload: p,
            }),
            Ok(_) => {}
            Err(e) => return fail(fmt, &format!("evaluation failed: {e}")),
        }
    }

    let stdout = io::stdout();
    if let Err(e) = output::render_lines(&mut stdout.lock(), &results, fmt) {
        return fail(fmt, &format!("render failed: {e}"));
    }
    ExitCode::SUCCESS
}

/// Like [`determine_input`] but preserves the full multi-line content (no trim)
/// so batch mode can split it into lines.
fn read_raw_input(cli: &Cli) -> Result<String, String> {
    if !io::stdin().is_terminal() {
        let mut buffer = String::new();
        io::stdin()
            .read_to_string(&mut buffer)
            .map_err(|e| format!("could not read stdin: {e}"))?;
        if buffer.trim().is_empty() {
            return Err("no input on stdin".into());
        }
        Ok(buffer)
    } else if let Some(text) = &cli.text {
        Ok(text.clone())
    } else {
        Err("no input: pipe text via stdin or pass it as an argument".into())
    }
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

/// Emit a command result (not analysis data) to stdout, honoring the format so
/// every subcommand is script-friendly. Human is a one-liner; json/ndjson is a
/// `{"status": ...}` object.
fn status(fmt: Format, code: &str, human: &str) -> ExitCode {
    match fmt {
        Format::Human => println!("ae: {human}"),
        Format::Json => println!("{}", serde_json::json!({ "status": code })),
        Format::Ndjson => println!("{}", serde_json::json!({ "status": code })),
    }
    ExitCode::SUCCESS
}

/// Report an error on stderr, as JSON when a machine format is in effect.
fn fail(fmt: Format, msg: &str) -> ExitCode {
    match fmt {
        Format::Human => eprintln!("ae: {msg}"),
        _ => eprintln!("{}", serde_json::json!({ "error": msg })),
    }
    ExitCode::FAILURE
}
