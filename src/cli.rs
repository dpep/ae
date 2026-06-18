//! Command-line surface: argument parsing, input resolution, role dispatch.
//!
//! stdout is reserved for data; all logging is routed to stderr so a consumer
//! piping `ae` always gets clean output.

use std::io::{self, IsTerminal, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{CommandFactory, Parser, Subcommand};

use crate::engine::Engine;
use crate::{ipc, output};

#[derive(Parser, Debug)]
#[command(name = "ae", author, version, about = "Acronym Engine")]
pub struct Cli {
    /// Dictionary management subcommand. Omit to analyze text (the default).
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Context text to scan. Optional when piping input via stdin.
    pub text: Option<String>,

    /// Start a detached background daemon (the warm Leader).
    #[arg(short, long)]
    pub daemon: bool,

    /// Stop the running background daemon.
    #[arg(long)]
    pub stop: bool,

    /// JSON output: a pretty object (analysis) or `{"status": …}` (commands).
    #[arg(short = 'j', long, global = true, conflicts_with = "ndjson")]
    pub json: bool,

    /// NDJSON output: one compact object per finding/line.
    #[arg(short = 'J', long, global = true)]
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

    /// Acronym dictionary database. Defaults to a per-user data dir
    /// (`$XDG_DATA_HOME/ae/acronyms.db`, else `~/.local/share/ae/acronyms.db`).
    #[arg(long, env = "AE_DB", global = true)]
    pub db: Option<PathBuf>,

    /// Embedding model: a path (directory or `.onnx` file) or a name resolved
    /// against the model search dirs. Defaults to the bundled/cached model.
    #[arg(short, long)]
    pub model: Option<String>,

    /// Emit engine telemetry to stderr.
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Internal: run as the Leader server process (spawned by `--daemon`).
    #[arg(long = "__serve", hide = true)]
    pub serve: bool,
}

impl Cli {
    /// Resolved dictionary path: `--db`/`$AE_DB` if given, else the per-user
    /// data-dir default.
    pub fn db_path(&self) -> PathBuf {
        self.db.clone().unwrap_or_else(default_db_path)
    }

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

/// Dictionary management commands. All honor `-j/-J` and operate on the `--db`
/// dictionary directly (no model needed).
#[derive(Subcommand, Debug)]
pub enum Command {
    /// List every acronym and expansion.
    List,
    /// Show the expansions of one acronym.
    Show { acronym: String },
    /// Search acronyms and expansions by substring.
    Search { query: String },
    /// Add an acronym/expansion to the dictionary.
    Add { acronym: String, expansion: String },
    /// Remove an acronym (all expansions), or one specific expansion.
    #[command(visible_alias = "delete")]
    Rm {
        acronym: String,
        expansion: Option<String>,
    },
}

/// Entry point — parse, set up logging, dispatch by role, render.
pub fn run() -> ExitCode {
    let cli = Cli::parse();
    init_logging(cli.verbose);
    let fmt = cli.format();

    // Dictionary management runs against the DB directly and exits.
    if let Some(command) = &cli.command {
        return run_command(command, &cli, fmt);
    }

    // Internal Leader process: serve until stopped, then exit.
    if cli.serve {
        return match ipc::serve(&cli.socket, &cli.db_path(), cli.model.as_deref()) {
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
        return match ipc::start_daemon(&cli.socket, &cli.db_path(), cli.model.as_deref()) {
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
    let engine = Engine::open(&cli.db_path(), cli.model.as_deref())?;
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
    let engine = match Engine::open(&cli.db_path(), cli.model.as_deref()) {
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

/// Run a dictionary management command against the `--db` store.
fn run_command(command: &Command, cli: &Cli, fmt: Format) -> ExitCode {
    let store = match crate::store::Store::open(&cli.db_path()) {
        Ok(store) => {
            let _ = store.seed_defaults();
            store
        }
        Err(e) => return fail(fmt, &format!("could not open dictionary: {e}")),
    };

    match command {
        Command::List => emit_entries(fmt, store.all_entries()),
        Command::Search { query } => emit_entries(fmt, store.search(query)),
        Command::Show { acronym } => {
            let entries = store.expansions_for(acronym).map(|rows| {
                rows.into_iter()
                    .map(|(_, e)| (acronym.to_uppercase(), e))
                    .collect()
            });
            emit_entries(fmt, entries)
        }
        Command::Add { acronym, expansion } => match store.add_entry(acronym, expansion) {
            Ok(_) => {
                let acronym = acronym.to_uppercase();
                match fmt {
                    Format::Human => println!("ae: added {acronym} → {expansion}"),
                    _ => println!(
                        "{}",
                        serde_json::json!({"status": "added", "acronym": acronym, "expansion": expansion})
                    ),
                }
                ExitCode::SUCCESS
            }
            Err(e) => fail(fmt, &format!("add failed: {e}")),
        },
        Command::Rm { acronym, expansion } => {
            let deleted = match expansion {
                Some(expansion) => store.delete_entry(acronym, expansion),
                None => store.delete_acronym(acronym),
            };
            match deleted {
                Ok(n) => {
                    let acronym = acronym.to_uppercase();
                    match fmt {
                        Format::Human => {
                            let noun = if n == 1 { "entry" } else { "entries" };
                            println!("ae: removed {n} {noun} for {acronym}");
                        }
                        _ => println!(
                            "{}",
                            serde_json::json!({"status": "removed", "acronym": acronym, "removed": n})
                        ),
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => fail(fmt, &format!("delete failed: {e}")),
            }
        }
    }
}

/// Render a list of `(acronym, expansion)` entries, or report the error.
fn emit_entries(fmt: Format, entries: rusqlite::Result<Vec<(String, String)>>) -> ExitCode {
    match entries {
        Ok(entries) => {
            let stdout = io::stdout();
            if let Err(e) = output::render_entries(&mut stdout.lock(), &entries, fmt) {
                return fail(fmt, &format!("render failed: {e}"));
            }
            ExitCode::SUCCESS
        }
        Err(e) => fail(fmt, &format!("dictionary error: {e}")),
    }
}

/// The default dictionary path under the per-user data dir.
fn default_db_path() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
        .unwrap_or_else(std::env::temp_dir);
    base.join("ae").join("acronyms.db")
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
