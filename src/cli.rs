//! Command-line surface: argument parsing, input resolution, role dispatch.
//!
//! stdout is reserved for data; all logging is routed to stderr so a consumer
//! piping `ae` always gets clean output.

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{CommandFactory, Parser, Subcommand};

use crate::engine::Engine;
use crate::{ipc, output};

const AFTER_HELP: &str = "\
Analyzing text returns three buckets:
  expansions   known acronyms, resolved from the dictionary and ranked
  extractions  acronyms defined inline now, e.g. KPI (Key Performance Indicator)
  candidates   acronym-shaped tokens seen but not yet resolved — to define later

Each expansion carries two scores: validity (is it a real expansion?) and
confidence (is it the meaning here?). -j/--json and -J/--ndjson switch to
machine output on every command.

A positional argument is analyzed as one text; piped stdin and --file are
streamed line by line, emitting line:col-tagged hits (use -J for a live stream).

Examples:
  ae \"ship the MVP this sprint\"     analyze one string
  cat access.log | ae -J             stream stdin line by line as NDJSON
  ae add OKR                         declare an acronym to watch & mine
  ae list perf                       list entries matching \"perf\"";

#[derive(Parser, Debug)]
#[command(
    name = "ae",
    author,
    version,
    about = "Extract and expand your org's acronyms — local and offline.",
    after_help = AFTER_HELP
)]
pub struct Cli {
    /// Dictionary management subcommand. Omit to analyze text (the default).
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Context text to scan. Optional when piping input via stdin.
    pub text: Option<String>,

    /// Route through a warm background daemon (faster for repeated calls),
    /// starting one if needed. With input, analyzes it too; with none, just
    /// starts the daemon.
    #[arg(short, long)]
    pub daemon: bool,

    /// Stop the running background daemon.
    #[arg(long)]
    pub stop: bool,

    /// Report whether the background daemon is running — with its version,
    /// embedder, and uptime. Read-only; never starts one. Exits non-zero when
    /// no daemon is up.
    #[arg(long)]
    pub status: bool,

    /// JSON output: a pretty object (analysis) or `{"status": …}` (commands).
    #[arg(short = 'j', long, global = true, conflicts_with = "ndjson")]
    pub json: bool,

    /// NDJSON output: one compact object per finding/line.
    #[arg(short = 'J', long, global = true)]
    pub ndjson: bool,

    /// Expand known acronyms only — don't extract or learn new ones (no writes).
    #[arg(short, long)]
    pub read_only: bool,

    /// Read input from this file, analyzed line by line (like piped stdin).
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

    /// Suppress normal stdout output (still does the work — e.g. silently learn).
    #[arg(short, long, global = true)]
    pub quiet: bool,

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
    /// List acronyms and expansions, optionally filtered by a substring of either.
    List { filter: Option<String> },
    /// Show the expansions of one acronym.
    Show { acronym: String },
    /// Add an acronym with one or more expansions. With no expansion, just
    /// declares the token as an acronym to watch — ae mines later text for
    /// what it expands to.
    Add {
        acronym: String,
        expansions: Vec<String>,
    },
    /// List candidate acronyms (seen but undefined) with provenance + watch
    /// state, by frequency.
    Candidates,
    /// Suggest speculative expansions mined from text, with confidence.
    /// Optionally for one acronym.
    Suggest {
        acronym: Option<String>,
        /// Hide suggestions below this confidence (default 0.30).
        #[arg(long)]
        min_confidence: Option<f32>,
        /// Keep at most this many suggestions per acronym.
        #[arg(short, long)]
        limit: Option<usize>,
    },
    /// Promote a candidate to the dictionary: add the given expansions, or pick
    /// from the mined suggestions interactively (fzf if available).
    Define {
        acronym: String,
        expansions: Vec<String>,
    },
    /// Garbage-collect speculation: merge prefix-duplicate expansions, drop
    /// low-confidence ones, and remove seen-once noise candidates.
    Prune {
        /// Drop expansions below this confidence (default 0.15).
        #[arg(long)]
        min_confidence: Option<f32>,
    },
    /// Remove an acronym. Bare `rm ACR` removes it when there's one expansion;
    /// otherwise pass a substring to pick one, or `--all` to remove every one.
    #[command(visible_alias = "delete")]
    Rm {
        acronym: String,
        /// Expansion substring selecting one variant when several exist.
        expansion: Option<String>,
        /// Remove every expansion of the acronym.
        #[arg(short, long)]
        all: bool,
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
            Ok(true) => status(fmt, cli.quiet, "stopped", "daemon stopped"),
            Ok(false) => status(fmt, cli.quiet, "not_running", "no daemon running"),
            Err(e) => fail(fmt, &format!("stop failed: {e}")),
        };
    }

    if cli.status {
        return run_status(&cli, fmt);
    }

    // A positional argument is a single text to analyze (one "blob"); piped
    // stdin and `--file` are streams analyzed line by line. `-d` also warms a
    // daemon alongside, for the single-text calls that tend to follow.
    if let Some(text) = cli.text.clone() {
        if cli.daemon
            && let Err(e) = ipc::start_daemon(&cli.socket, &cli.db_path(), cli.model.as_deref())
        {
            log::debug!("daemon unavailable ({e}); evaluating in-process");
        }
        return serve_text(&cli, fmt, &text);
    }

    // `--file` streams a file line by line.
    if let Some(path) = cli.file.clone() {
        let reader = match std::fs::File::open(&path) {
            Ok(f) => io::BufReader::new(f),
            Err(e) => return fail(fmt, &format!("could not read {}: {e}", path.display())),
        };
        if cli.daemon {
            let _ = ipc::start_daemon(&cli.socket, &cli.db_path(), cli.model.as_deref());
        }
        return run_stream(&cli, fmt, Box::new(reader));
    }

    // Piped stdin streams line by line. An empty pipe (or `/dev/null`) has
    // nothing to stream, so with `-d` we fall through to (start and) report the
    // daemon rather than printing an empty result.
    if !io::stdin().is_terminal() {
        let mut reader = io::stdin().lock();
        let empty = matches!(reader.fill_buf(), Ok(b) if b.is_empty());
        if !(empty && cli.daemon) {
            if cli.daemon {
                let _ = ipc::start_daemon(&cli.socket, &cli.db_path(), cli.model.as_deref());
            }
            return run_stream(&cli, fmt, Box::new(reader));
        }
    }

    // Nothing to analyze: `-d` (ensures and) reports the daemon; otherwise a
    // bare invocation shows help rather than erroring.
    if cli.daemon {
        return match ipc::start_daemon(&cli.socket, &cli.db_path(), cli.model.as_deref()) {
            Ok(ipc::DaemonOutcome::Started) => status(fmt, cli.quiet, "started", "daemon started"),
            Ok(ipc::DaemonOutcome::AlreadyRunning) => {
                status(fmt, cli.quiet, "already_running", "daemon already running")
            }
            Err(e) => fail(fmt, &format!("could not start daemon: {e}")),
        };
    }
    let _ = Cli::command().print_help();
    println!();
    ExitCode::SUCCESS
}

/// Serve one chunk of text: proxy to the daemon if one is up, else self-heal by
/// evaluating in-process, then render the result.
fn serve_text(cli: &Cli, fmt: Format, text: &str) -> ExitCode {
    let payload = match ipc::run_follower(&cli.socket, text, cli.read_only) {
        Ok(p) => {
            log::debug!("served by daemon");
            p
        }
        Err(_) => {
            log::debug!("no daemon; evaluating in-process");
            match evaluate_in_process(cli, text) {
                Ok(p) => p,
                Err(e) => return fail(fmt, &format!("evaluation failed: {e}")),
            }
        }
    };
    if !cli.quiet {
        let stdout = io::stdout();
        if let Err(e) = output::render(&mut stdout.lock(), &payload, fmt) {
            return fail(fmt, &format!("render failed: {e}"));
        }
    }
    ExitCode::SUCCESS
}

/// The self-healing fallback: open the shared persistent engine and evaluate.
/// Honors `--read-only` (expand without learning).
fn evaluate_in_process(cli: &Cli, text: &str) -> rusqlite::Result<crate::types::AnalysisPayload> {
    let engine = Engine::open(&cli.db_path(), cli.model.as_deref())?;
    if cli.read_only {
        return engine.expand_only(text);
    }
    let payload = engine.analyze(text)?;
    // Amortized consolidation on a cadence (best-effort).
    let _ = engine.consolidate_if_due(
        crate::store::PRUNE_MIN_CONFIDENCE,
        crate::engine::prune_grace_secs(),
    );
    Ok(payload)
}

/// Stream input line by line through one warm in-process engine, emitting each
/// line's findings as it's analyzed. Human/NDJSON flush per line (so `tail -f |
/// ae -J` is live); pretty JSON can't emit a partial array, so it aggregates and
/// renders once at the end. The default path for piped stdin and `--file`.
/// In-process (not via the daemon), since it's one pass over many lines.
fn run_stream(cli: &Cli, fmt: Format, reader: Box<dyn io::BufRead>) -> ExitCode {
    let engine = match Engine::open(&cli.db_path(), cli.model.as_deref()) {
        Ok(e) => e,
        Err(e) => return fail(fmt, &format!("could not open engine: {e}")),
    };

    // Pretty JSON needs the whole array to be valid, so it buffers; human and
    // NDJSON stream per line.
    let buffered = fmt == Format::Json;
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut collected: Vec<output::LineResult> = Vec::new();
    let mut emitted = 0usize;

    for (i, line) in reader.lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(e) => return fail(fmt, &format!("could not read input: {e}")),
        };
        if line.trim().is_empty() {
            continue;
        }
        let payload = if cli.read_only {
            engine.expand_only(&line)
        } else {
            engine.analyze(&line)
        };
        let payload = match payload {
            Ok(p) => p,
            // Skip-and-continue so one bad line doesn't abort a long stream.
            Err(e) => {
                log::warn!("line {}: {e}", i + 1);
                continue;
            }
        };
        if payload.is_empty() {
            continue;
        }
        let result = output::LineResult {
            line: i + 1,
            text: line,
            payload,
        };
        if cli.quiet {
            continue; // work (learning) still happened; just don't emit
        }
        if buffered {
            collected.push(result);
        } else if let Err(e) = output::stream_line(&mut out, &result, fmt) {
            return fail(fmt, &format!("render failed: {e}"));
        } else {
            emitted += 1;
        }
    }

    if !cli.quiet {
        if buffered {
            if let Err(e) = output::render_lines(&mut out, &collected, fmt) {
                return fail(fmt, &format!("render failed: {e}"));
            }
        } else if emitted == 0 && fmt == Format::Human {
            let _ = writeln!(out, "No acronyms found.");
        }
    }
    ExitCode::SUCCESS
}

/// Report daemon status to stdout, honoring the format. Read-only — probes the
/// socket without starting anything. Exits non-zero when no daemon is up, so it
/// doubles as a health check (`ae --status && …`).
fn run_status(cli: &Cli, fmt: Format) -> ExitCode {
    let report = ipc::status(&cli.socket).unwrap_or(None);
    if !cli.quiet {
        let stdout = io::stdout();
        if let Err(e) = output::render_status(
            &mut stdout.lock(),
            report.as_ref(),
            &cli.socket,
            &cli.db_path(),
            fmt,
        ) {
            return fail(fmt, &format!("render failed: {e}"));
        }
    }
    if report.is_some() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
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

    let quiet = cli.quiet;
    match command {
        // Read-only commands produce only output, so `--quiet` is a no-op exit.
        Command::List { filter } if quiet => {
            let _ = filter;
            ExitCode::SUCCESS
        }
        Command::List { filter } => {
            let entries = match filter {
                Some(f) => store.search(f),
                None => store.all_entries(),
            };
            emit_entries(fmt, entries)
        }
        Command::Show { .. } | Command::Candidates | Command::Suggest { .. } if quiet => {
            ExitCode::SUCCESS
        }
        Command::Show { acronym } => {
            let entries = store.expansions_for(acronym).map(|rows| {
                rows.into_iter()
                    .map(|(_, e, source)| (acronym.to_uppercase(), e, source))
                    .collect()
            });
            emit_entries(fmt, entries)
        }
        Command::Candidates => match store.candidates_detailed() {
            Ok(rows) => {
                // (acronym, count, source, on watch list)
                let rows: Vec<(String, i64, String, bool)> = rows
                    .into_iter()
                    .map(|(a, n, source)| {
                        let watching = source == "declared" || n >= crate::store::WATCH_THRESHOLD;
                        (a, n, source, watching)
                    })
                    .collect();
                let stdout = io::stdout();
                if let Err(e) = output::render_candidates(&mut stdout.lock(), &rows, fmt) {
                    return fail(fmt, &format!("render failed: {e}"));
                }
                ExitCode::SUCCESS
            }
            Err(e) => fail(fmt, &format!("dictionary error: {e}")),
        },
        Command::Suggest {
            acronym,
            min_confidence,
            limit,
        } => run_suggest(
            &store,
            fmt,
            acronym.as_deref(),
            min_confidence.unwrap_or(SUGGEST_MIN_CONFIDENCE),
            *limit,
        ),
        Command::Add {
            acronym,
            expansions,
        } => run_add(&store, fmt, quiet, acronym, expansions),
        Command::Define {
            acronym,
            expansions,
        } => run_define(&store, fmt, quiet, acronym, expansions),
        Command::Prune { min_confidence } => run_prune(
            &store,
            fmt,
            quiet,
            min_confidence.unwrap_or(crate::store::PRUNE_MIN_CONFIDENCE),
        ),
        Command::Rm {
            acronym,
            expansion,
            all,
        } => run_rm(&store, fmt, quiet, acronym, expansion.as_deref(), *all),
    }
}

/// Add one acronym with one or more expansions.
fn run_add(
    store: &crate::store::Store,
    fmt: Format,
    quiet: bool,
    acronym: &str,
    expansions: &[String],
) -> ExitCode {
    // No expansion → declare it's an acronym to watch ("figure it out").
    if expansions.is_empty() {
        return match store.declare_acronym(acronym) {
            Ok(()) => status(
                fmt,
                quiet,
                "watching",
                &format!("now watching {} for expansions", acronym.to_uppercase()),
            ),
            Err(e) => fail(fmt, &format!("watch failed: {e}")),
        };
    }
    for expansion in expansions {
        if let Err(e) = store.add_entry(acronym, expansion, "user") {
            return fail(fmt, &format!("add failed: {e}"));
        }
    }
    let acronym = acronym.to_uppercase();
    if !quiet {
        match fmt {
            Format::Human => {
                for expansion in expansions {
                    println!("ae: added {acronym} → {expansion}");
                }
            }
            _ => println!(
                "{}",
                serde_json::json!({"status": "added", "acronym": acronym, "added": expansions})
            ),
        }
    }
    ExitCode::SUCCESS
}

/// `suggest` hides anything below this (keep the bar high — speculation is
/// noisy); `prune`/GC use the gentler `store::PRUNE_MIN_CONFIDENCE`.
const SUGGEST_MIN_CONFIDENCE: f32 = 0.30;

/// Score speculative expansions: `(acronym, expansion, count, confidence)`,
/// for one acronym or all. Shared by `suggest`, `define`, and `prune`.
fn score_potentials(
    store: &crate::store::Store,
    acronym: Option<&str>,
) -> rusqlite::Result<Vec<(String, String, i64, f32)>> {
    let raw: Vec<(String, String, i64, f64)> = match acronym {
        Some(acr) => store
            .potentials_for(acr)?
            .into_iter()
            .map(|(exp, n, coh)| (acr.to_uppercase(), exp, n, coh))
            .collect(),
        None => store.all_potentials()?,
    };
    let mut totals: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    for (acr, _, n, _) in &raw {
        *totals.entry(acr.clone()).or_insert(0) += n;
    }
    Ok(raw
        .into_iter()
        .map(|(acr, exp, n, coh)| {
            let conf = crate::store::confidence(n, coh, totals[&acr]);
            (acr, exp, n, conf)
        })
        .collect())
}

/// Render speculative expansions at or above `min` confidence, best first, at
/// most `limit` per acronym.
fn run_suggest(
    store: &crate::store::Store,
    fmt: Format,
    acronym: Option<&str>,
    min: f32,
    limit: Option<usize>,
) -> ExitCode {
    let mut scored = match score_potentials(store, acronym) {
        Ok(s) => s,
        Err(e) => return fail(fmt, &format!("dictionary error: {e}")),
    };
    scored.retain(|(_, _, _, conf)| *conf >= min);
    scored.sort_by(|a, b| a.0.cmp(&b.0).then(b.3.total_cmp(&a.3)));

    if let Some(limit) = limit {
        let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        scored.retain(|(acr, _, _, _)| {
            let n = seen.entry(acr.clone()).or_insert(0);
            *n += 1;
            *n <= limit
        });
    }

    let stdout = io::stdout();
    if let Err(e) = output::render_suggestions(&mut stdout.lock(), &scored, fmt) {
        return fail(fmt, &format!("render failed: {e}"));
    }
    ExitCode::SUCCESS
}

/// GC speculation: spell-correct mined expansions, dedup (prefix + fuzzy), drop
/// low-confidence ones, and remove seen-once noise candidates.
fn run_prune(store: &crate::store::Store, fmt: Format, quiet: bool, min: f32) -> ExitCode {
    match store.consolidate(min, crate::engine::prune_grace_secs()) {
        Ok(s) => {
            if !quiet {
                match fmt {
                    Format::Human => println!(
                        "ae: pruned — corrected {}, merged {}, dropped {} low-confidence, removed {} noise candidates",
                        s.corrected, s.merged, s.dropped, s.candidates
                    ),
                    _ => println!(
                        "{}",
                        serde_json::json!({"status": "pruned", "corrected": s.corrected, "merged": s.merged, "dropped": s.dropped, "candidates_removed": s.candidates})
                    ),
                }
            }
            ExitCode::SUCCESS
        }
        Err(e) => fail(fmt, &format!("prune failed: {e}")),
    }
}

/// Promote a candidate: add the given expansions, or pick interactively from the
/// mined suggestions (all of them — diagnostic — via fzf or a numbered prompt).
fn run_define(
    store: &crate::store::Store,
    fmt: Format,
    quiet: bool,
    acronym: &str,
    expansions: &[String],
) -> ExitCode {
    let chosen: Vec<String> = if !expansions.is_empty() {
        expansions.to_vec()
    } else {
        let mut suggestions = match score_potentials(store, Some(acronym)) {
            Ok(s) => s,
            Err(e) => return fail(fmt, &format!("dictionary error: {e}")),
        };
        if suggestions.is_empty() {
            return fail(
                fmt,
                &format!(
                    "no suggestions for {} — add one explicitly",
                    acronym.to_uppercase()
                ),
            );
        }
        // Show *all* candidate expansions when choosing — they're diagnostic.
        suggestions.sort_by(|a, b| b.3.total_cmp(&a.3));
        let ranked: Vec<(String, f32)> = suggestions
            .into_iter()
            .map(|(_, exp, _, conf)| (exp, conf))
            .collect();
        match pick_expansions(&acronym.to_uppercase(), &ranked) {
            Some(sel) if !sel.is_empty() => sel,
            Some(_) => return status(fmt, quiet, "cancelled", "nothing selected"),
            None => return ExitCode::FAILURE,
        }
    };

    let acronym_upper = acronym.to_uppercase();
    for expansion in &chosen {
        if let Err(e) = store.add_entry(acronym, expansion, "user") {
            return fail(fmt, &format!("add failed: {e}"));
        }
    }
    if !quiet {
        match fmt {
            Format::Human => {
                for expansion in &chosen {
                    println!("ae: added {acronym_upper} → {expansion}");
                }
            }
            _ => println!(
                "{}",
                serde_json::json!({"status": "defined", "acronym": acronym_upper, "added": chosen})
            ),
        }
    }
    ExitCode::SUCCESS
}

/// Let the user pick one or more expansions interactively. Requires a TTY;
/// prefers `fzf` (with multi-select), falls back to a numbered prompt.
fn pick_expansions(acronym: &str, suggestions: &[(String, f32)]) -> Option<Vec<String>> {
    if !(io::stdin().is_terminal() && io::stderr().is_terminal()) {
        eprintln!("ae: not a TTY — pass expansions explicitly: ae define {acronym} \"...\"");
        return None;
    }
    match pick_with_fzf(acronym, suggestions) {
        Some(sel) => Some(sel),
        None => pick_numbered(acronym, suggestions),
    }
}

/// Multi-select via `fzf`. `None` if fzf isn't available (caller falls back).
fn pick_with_fzf(acronym: &str, suggestions: &[(String, f32)]) -> Option<Vec<String>> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut child = Command::new("fzf")
        .args(["--multi", "--with-nth=1", "--delimiter=\t"])
        .arg(format!("--prompt={acronym}> "))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .ok()?;

    if let Some(mut stdin) = child.stdin.take() {
        for (expansion, conf) in suggestions {
            let _ = writeln!(stdin, "{expansion}\t({conf:.2})");
        }
    }
    let out = child.wait_with_output().ok()?;
    // Non-zero exit means the user cancelled — selected nothing.
    let selected = if out.status.success() {
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| l.split('\t').next())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect()
    } else {
        Vec::new()
    };
    Some(selected)
}

/// Numbered-prompt fallback when `fzf` isn't installed.
fn pick_numbered(acronym: &str, suggestions: &[(String, f32)]) -> Option<Vec<String>> {
    use std::io::Write;
    let mut err = io::stderr();
    let _ = writeln!(err, "Suggestions for {acronym}:");
    for (i, (expansion, conf)) in suggestions.iter().enumerate() {
        let _ = writeln!(err, "  {}) {expansion}  ({conf:.2})", i + 1);
    }
    let _ = write!(err, "select (e.g. 1 3, blank to cancel): ");
    let _ = err.flush();

    let mut line = String::new();
    io::stdin().read_line(&mut line).ok()?;
    let chosen = line
        .split_whitespace()
        .filter_map(|t| t.parse::<usize>().ok())
        .filter_map(|n| suggestions.get(n.wrapping_sub(1)))
        .map(|(expansion, _)| expansion.clone())
        .collect();
    Some(chosen)
}

/// Resolve and execute a removal, disambiguating among multiple expansions.
fn run_rm(
    store: &crate::store::Store,
    fmt: Format,
    quiet: bool,
    acronym: &str,
    pattern: Option<&str>,
    all: bool,
) -> ExitCode {
    let variants = match store.expansions_for(acronym) {
        Ok(v) => v,
        Err(e) => return fail(fmt, &format!("delete failed: {e}")),
    };
    let acronym = acronym.to_uppercase();
    if variants.is_empty() {
        return removed_status(fmt, quiet, &acronym, 0);
    }

    let all_exps: Vec<String> = variants.iter().map(|(_, e, _)| e.clone()).collect();
    let targets: Vec<String> = if all {
        all_exps
    } else if let Some(pat) = pattern {
        let needle = pat.to_lowercase();
        let matched: Vec<String> = all_exps
            .iter()
            .filter(|e| e.to_lowercase().contains(&needle))
            .cloned()
            .collect();
        match matched.len() {
            0 => {
                return refuse(
                    fmt,
                    &acronym,
                    &format!("no expansion matches \"{pat}\""),
                    &all_exps,
                );
            }
            1 => matched,
            _ => {
                return refuse(
                    fmt,
                    &acronym,
                    &format!("\"{pat}\" matches several expansions"),
                    &matched,
                );
            }
        }
    } else if all_exps.len() == 1 {
        all_exps
    } else {
        return refuse(fmt, &acronym, "has several expansions", &all_exps);
    };

    let mut removed = 0;
    for expansion in &targets {
        match store.delete_entry(&acronym, expansion) {
            Ok(n) => removed += n,
            Err(e) => return fail(fmt, &format!("delete failed: {e}")),
        }
    }
    removed_status(fmt, quiet, &acronym, removed)
}

fn removed_status(fmt: Format, quiet: bool, acronym: &str, n: usize) -> ExitCode {
    if !quiet {
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
    }
    ExitCode::SUCCESS
}

/// Decline an ambiguous removal, listing the expansions so the user can refine.
fn refuse(fmt: Format, acronym: &str, reason: &str, expansions: &[String]) -> ExitCode {
    match fmt {
        Format::Human => {
            eprintln!("ae: {acronym} {reason} — specify a substring or use --all:");
            for e in expansions {
                eprintln!("  {e}");
            }
        }
        _ => eprintln!(
            "{}",
            serde_json::json!({"error": "ambiguous", "acronym": acronym, "reason": reason, "expansions": expansions})
        ),
    }
    ExitCode::FAILURE
}

/// Render a list of `(acronym, expansion, source)` entries, or report the error.
fn emit_entries(fmt: Format, entries: rusqlite::Result<Vec<(String, String, String)>>) -> ExitCode {
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
fn status(fmt: Format, quiet: bool, code: &str, human: &str) -> ExitCode {
    if !quiet {
        match fmt {
            Format::Human => println!("ae: {human}"),
            _ => println!("{}", serde_json::json!({ "status": code })),
        }
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
