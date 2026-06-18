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
    /// List candidate acronyms (seen but undefined) by frequency.
    Candidates,
    /// Suggest speculative expansions mined from text, with confidence.
    /// Optionally for one acronym.
    Suggest {
        acronym: Option<String>,
        /// Hide suggestions below this confidence (default 0.15).
        #[arg(long)]
        min_confidence: Option<f32>,
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
        Command::Candidates => match store.candidates() {
            Ok(candidates) => {
                let stdout = io::stdout();
                if let Err(e) = output::render_candidates(&mut stdout.lock(), &candidates, fmt) {
                    return fail(fmt, &format!("render failed: {e}"));
                }
                ExitCode::SUCCESS
            }
            Err(e) => fail(fmt, &format!("dictionary error: {e}")),
        },
        Command::Suggest {
            acronym,
            min_confidence,
        } => run_suggest(
            &store,
            fmt,
            acronym.as_deref(),
            min_confidence.unwrap_or(MIN_CONFIDENCE),
        ),
        Command::Define {
            acronym,
            expansions,
        } => run_define(&store, fmt, acronym, expansions),
        Command::Prune { min_confidence } => {
            run_prune(&store, fmt, min_confidence.unwrap_or(MIN_CONFIDENCE))
        }
        Command::Rm {
            acronym,
            expansion,
            all,
        } => run_rm(&store, fmt, acronym, expansion.as_deref(), *all),
    }
}

/// Default confidence floor below which speculative suggestions are hidden and
/// pruning discards them.
const MIN_CONFIDENCE: f32 = 0.15;

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
            let conf = confidence(n, coh, totals[&acr]);
            (acr, exp, n, conf)
        })
        .collect())
}

/// Render speculative expansions at or above `min` confidence, best first.
fn run_suggest(
    store: &crate::store::Store,
    fmt: Format,
    acronym: Option<&str>,
    min: f32,
) -> ExitCode {
    let mut scored = match score_potentials(store, acronym) {
        Ok(s) => s,
        Err(e) => return fail(fmt, &format!("dictionary error: {e}")),
    };
    scored.retain(|(_, _, _, conf)| *conf >= min);
    scored.sort_by(|a, b| a.0.cmp(&b.0).then(b.3.total_cmp(&a.3)));

    let stdout = io::stdout();
    if let Err(e) = output::render_suggestions(&mut stdout.lock(), &scored, fmt) {
        return fail(fmt, &format!("render failed: {e}"));
    }
    ExitCode::SUCCESS
}

/// Blend recurrence and coherence into a `[0, 1]` confidence: `share` (this
/// expansion's fraction of the acronym's sightings) and `mean_coh` (average
/// context coherence) weighted equally, then damped so a lone sighting can't
/// reach certainty.
fn confidence(count: i64, coh_sum: f64, total: i64) -> f32 {
    let count = count.max(1) as f32;
    let share = count / total.max(1) as f32;
    let mean_coh = (coh_sum as f32 / count).clamp(0.0, 1.0);
    ((0.5 * share + 0.5 * mean_coh) * (count / (count + 1.0))).clamp(0.0, 1.0)
}

/// GC speculation: dedup prefix-duplicate expansions, drop low-confidence ones,
/// and remove seen-once noise candidates.
fn run_prune(store: &crate::store::Store, fmt: Format, min: f32) -> ExitCode {
    let result = (|| -> rusqlite::Result<(usize, usize, usize)> {
        let mut merged = 0;
        for acronym in store.distinct_potential_acronyms()? {
            merged += store.dedup_potentials(&acronym)?;
        }
        let mut dropped = 0;
        for (acronym, expansion, _, conf) in score_potentials(store, None)? {
            if conf < min {
                dropped += store.delete_potential(&acronym, &expansion)?;
            }
        }
        let candidates = store.prune_noise_candidates()?;
        Ok((merged, dropped, candidates))
    })();

    match result {
        Ok((merged, dropped, candidates)) => {
            match fmt {
                Format::Human => println!(
                    "ae: pruned — merged {merged}, dropped {dropped} low-confidence, removed {candidates} noise candidates"
                ),
                _ => println!(
                    "{}",
                    serde_json::json!({"status": "pruned", "merged": merged, "dropped": dropped, "candidates_removed": candidates})
                ),
            }
            ExitCode::SUCCESS
        }
        Err(e) => fail(fmt, &format!("prune failed: {e}")),
    }
}

/// Promote a candidate: add the given expansions, or pick interactively from the
/// mined suggestions (fzf if available, else a numbered prompt).
fn run_define(
    store: &crate::store::Store,
    fmt: Format,
    acronym: &str,
    expansions: &[String],
) -> ExitCode {
    let chosen: Vec<String> = if !expansions.is_empty() {
        expansions.to_vec()
    } else {
        let suggestions = match score_potentials(store, Some(acronym)) {
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
        let mut ranked: Vec<(String, f32)> = suggestions
            .into_iter()
            .map(|(_, exp, _, conf)| (exp, conf))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        match pick_expansions(&acronym.to_uppercase(), &ranked) {
            Some(sel) if !sel.is_empty() => sel,
            Some(_) => return status(fmt, "cancelled", "nothing selected"),
            None => return ExitCode::FAILURE,
        }
    };

    let acronym_upper = acronym.to_uppercase();
    for expansion in &chosen {
        if let Err(e) = store.add_entry(acronym, expansion) {
            return fail(fmt, &format!("add failed: {e}"));
        }
    }
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
        return removed_status(fmt, &acronym, 0);
    }

    let all_exps: Vec<String> = variants.iter().map(|(_, e)| e.clone()).collect();
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
    removed_status(fmt, &acronym, removed)
}

fn removed_status(fmt: Format, acronym: &str, n: usize) -> ExitCode {
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

#[cfg(test)]
mod tests {
    use super::confidence;

    #[test]
    fn confidence_rewards_recurrence_and_coherence() {
        // Dominant, recurring, on-topic expansion vs a lone off-topic match.
        let strong = confidence(4, 4.0, 5); // share 0.8, coherence 1.0
        let weak = confidence(1, 0.1, 5); // share 0.2, coherence 0.1
        assert!(strong > weak);
        assert!((0.0..=1.0).contains(&strong));
        assert!((0.0..=1.0).contains(&weak));
    }
}
