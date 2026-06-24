//! End-to-end tests that drive the built `ae` binary with an isolated socket
//! and DB, exercising the no-daemon self-healing fallback path.
//!
//! A single text is analyzed by passing it as the positional argument (the
//! "blob" path that yields a rich `AnalysisPayload`). Piped stdin is the
//! streaming, line-by-line path — exercised via [`run_piped`].

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// A unique socket path per test, so the derived `.db`/`.lock` are isolated and
/// tests can run in parallel.
fn scratch_socket(label: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("ae-e2e-{}-{label}.sock", std::process::id()));
    for ext in ["sock", "db", "db-wal", "db-shm", "lock"] {
        let _ = std::fs::remove_file(p.with_extension(ext));
    }
    p
}

struct Output {
    success: bool,
    stdout: String,
    stderr: String,
}

/// Run `ae` with `args`; if `text` is given it's the positional argument — the
/// single-text "blob" path. Auto-consolidation is disabled so tests are
/// deterministic.
fn run(socket: &std::path::Path, args: &[&str], text: Option<&str>) -> Output {
    run_with_env(socket, args, text, &[("AE_CONSOLIDATE_SECS", "-1")])
}

/// Like [`run`], but with explicit env overrides — e.g. forcing GC on.
fn run_with_env(
    socket: &std::path::Path,
    args: &[&str],
    text: Option<&str>,
    env: &[(&str, &str)],
) -> Output {
    let mut argv: Vec<&str> = args.to_vec();
    if let Some(t) = text {
        argv.push(t);
    }
    exec(socket, &argv, None, env)
}

/// Pipe `stdin` into `ae` — the streaming, line-by-line input path.
fn run_piped(socket: &std::path::Path, args: &[&str], stdin: &str) -> Output {
    exec(socket, args, Some(stdin), &[("AE_CONSOLIDATE_SECS", "-1")])
}

/// Spawn the built binary with an isolated socket/DB, optionally writing
/// `stdin`, and capture its output.
fn exec(
    socket: &std::path::Path,
    args: &[&str],
    stdin: Option<&str>,
    env: &[(&str, &str)],
) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_ae"));
    cmd.arg("--socket")
        .arg(socket)
        .arg("--db")
        .arg(socket.with_extension("db"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().unwrap();
    if let Some(text) = stdin {
        child
            .stdin
            .take()
            .unwrap()
            .write_all(text.as_bytes())
            .unwrap();
    } // dropping the handle closes stdin (EOF) either way
    let out = child.wait_with_output().unwrap();
    Output {
        success: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

#[test]
fn pipes_text_and_expands_a_known_acronym() {
    let sock = scratch_socket("human");
    let out = run(&sock, &[], Some("Check the OKR board today."));
    assert!(out.success, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Objectives and Key Results"));
    assert!(out.stdout.contains("expansion"));
}

#[test]
fn json_output_parses_and_reports_learning() {
    let sock = scratch_socket("json");
    let out = run(
        &sock,
        &["-j"],
        Some("Our KPI (Key Performance Indicator) is up."),
    );
    assert!(out.success, "stderr: {}", out.stderr);
    let v: Vec<serde_json::Value> = serde_json::from_str(&out.stdout).unwrap();
    assert!(
        v.iter()
            .any(|f| f["kind"] == "extraction" && f["acronym"] == "KPI")
    );
}

#[test]
fn ndjson_emits_one_object_per_line() {
    let sock = scratch_socket("ndjson");
    let out = run(&sock, &["-J"], Some("The OKR review is Friday."));
    assert!(out.success, "stderr: {}", out.stderr);
    let mut lines = out.stdout.lines();
    let first: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    assert_eq!(first["kind"], "expansion");
    assert_eq!(first["acronym"], "OKR");
}

#[test]
fn learning_persists_across_invocations_via_shared_db() {
    let sock = scratch_socket("persist");
    // First pass teaches ZQ.
    let first = run(&sock, &[], Some("The ZQ (Zebra Queue) is deep."));
    assert!(first.stdout.contains("extraction"));
    // Second pass — same socket → same DB — now expands it.
    let second = run(&sock, &["-j"], Some("Drain the ZQ now."));
    let v: Vec<serde_json::Value> = serde_json::from_str(&second.stdout).unwrap();
    assert!(
        v.iter().any(|f| f["kind"] == "expansion"
            && f["acronym"] == "ZQ"
            && f["expansion"] == "Zebra Queue"),
        "ZQ was not expanded on the second pass: {}",
        second.stdout
    );
}

#[test]
fn model_flag_is_accepted_and_degrades_gracefully() {
    let sock = scratch_socket("model");
    // A bogus model path must not break the run — it falls back to the hash
    // embedder, and expansion (trie + dictionary) is unaffected.
    let out = run(
        &sock,
        &["--model", "/no/such/model", "-j"],
        Some("Check the OKR board."),
    );
    assert!(out.success, "stderr: {}", out.stderr);
    let v: Vec<serde_json::Value> = serde_json::from_str(&out.stdout).unwrap();
    assert!(
        v.iter()
            .any(|f| f["kind"] == "expansion" && f["acronym"] == "OKR")
    );
    // The fallback is announced once, clearly, on stderr so it's fixable — and
    // never pollutes stdout (which stays clean JSON, parsed above).
    assert!(
        out.stderr.contains("hash fallback"),
        "stderr: {}",
        out.stderr
    );
}

#[test]
fn read_only_expands_without_learning_or_persisting() {
    let sock = scratch_socket("readonly");
    // Read-only: inline definition present, but nothing is learned...
    let first = run(
        &sock,
        &["--read-only", "-j"],
        Some("The ZQ (Zebra Queue) backs the OKR."),
    );
    let v: Vec<serde_json::Value> = serde_json::from_str(&first.stdout).unwrap();
    assert!(
        v.iter()
            .any(|f| f["kind"] == "expansion" && f["acronym"] == "OKR")
    );
    assert!(!v.iter().any(|f| f["kind"] == "extraction"));

    // ...and nothing was persisted: a normal later pass still doesn't know ZQ.
    let second = run(&sock, &["-j"], Some("Drain the ZQ."));
    let v2: Vec<serde_json::Value> = serde_json::from_str(&second.stdout).unwrap();
    assert!(
        !v2.iter()
            .any(|f| f["kind"] == "expansion" && f["acronym"] == "ZQ")
    );
}

#[test]
fn short_json_and_ndjson_flags_select_the_format() {
    let sock = scratch_socket("shortflags");
    // -j → a pretty JSON array of findings.
    let j = run(&sock, &["-j"], Some("Check the OKR board."));
    serde_json::from_str::<Vec<serde_json::Value>>(&j.stdout).expect("-j is a JSON array");

    // -J → one compact object per line.
    let nd = run(&sock, &["-J"], Some("Check the OKR board."));
    let line = nd.stdout.lines().next().unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(line).unwrap()["kind"],
        "expansion"
    );
}

#[test]
fn commands_emit_status_json_in_machine_mode() {
    let sock = scratch_socket("status");
    // No daemon running → stop reports it as a JSON status object on stdout.
    let out = run(&sock, &["--stop", "-j"], None);
    assert!(out.success);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["status"], "not_running");
}

#[test]
fn piped_stdin_streams_findings_as_an_array() {
    let sock = scratch_socket("stream");
    let input = "first line has an OKR\nsecond mentions the API\n";
    // Pretty JSON aggregates the streamed findings into one array.
    let out = run_piped(&sock, &["-j"], input);
    assert!(out.success, "stderr: {}", out.stderr);
    let hits: Vec<serde_json::Value> = serde_json::from_str(&out.stdout).unwrap();
    assert!(hits.iter().any(|h| h["acronym"] == "OKR"));
    assert!(hits.iter().any(|h| h["acronym"] == "API"));
}

#[test]
fn piped_ndjson_emits_a_finding_object_per_line() {
    let sock = scratch_socket("streamnd");
    let out = run_piped(&sock, &["-J"], "the OKR here\nand the API there\n");
    assert!(out.success, "stderr: {}", out.stderr);
    let hits: Vec<serde_json::Value> = out
        .stdout
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(hits.iter().any(|v| v["acronym"] == "OKR"));
    assert!(hits.iter().any(|v| v["acronym"] == "API"));
}

#[test]
fn arg_and_pipe_emit_the_same_shape() {
    let sock = scratch_socket("dispatch");
    // A positional blob and a piped line of the same text now produce an
    // identical flat array of findings — no "sentence"/"line" wrapper either way.
    let blob = run(&sock, &["-j"], Some("the OKR review"));
    let bv: Vec<serde_json::Value> = serde_json::from_str(&blob.stdout).unwrap();
    let stream = run_piped(&sock, &["-j"], "the OKR review\n");
    let sv: Vec<serde_json::Value> = serde_json::from_str(&stream.stdout).unwrap();
    assert_eq!(bv, sv);
    assert!(
        bv.iter()
            .all(|f| f["kind"].is_string() && f.get("line").is_none())
    );
}

#[test]
fn file_flag_reads_a_file_line_by_line() {
    let sock = scratch_socket("file");
    let path = std::env::temp_dir().join(format!("ae-input-{}.txt", std::process::id()));
    std::fs::write(&path, "intro line\nthis row has the OKR\n").unwrap();
    // The file alone drives the line-by-line path (like piped stdin).
    let out = run(&sock, &["--file", path.to_str().unwrap(), "-j"], None);
    assert!(out.success, "stderr: {}", out.stderr);
    let hits: Vec<serde_json::Value> = serde_json::from_str(&out.stdout).unwrap();
    assert!(hits.iter().any(|h| h["acronym"] == "OKR"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn piped_human_output_lists_findings() {
    let sock = scratch_socket("streamhuman");
    let out = run_piped(&sock, &[], "line one\nthe OKR is here\n");
    assert!(out.success, "stderr: {}", out.stderr);
    assert!(out.stdout.lines().any(|l| l.contains("OKR")));
}

#[test]
fn unknown_acronyms_are_surfaced() {
    let sock = scratch_socket("unknown");
    let out = run(&sock, &["-j"], Some("hi there MVP"));
    assert!(out.success, "stderr: {}", out.stderr);
    let v: Vec<serde_json::Value> = serde_json::from_str(&out.stdout).unwrap();
    assert!(
        v.iter()
            .any(|f| f["kind"] == "candidate" && f["acronym"] == "MVP"),
        "MVP not surfaced: {}",
        out.stdout
    );
}

#[test]
fn manage_add_list_search_show_then_remove() {
    let sock = scratch_socket("manage");

    assert!(run(&sock, &["add", "MVP", "Minimum Viable Product"], None).success);

    let list = run(&sock, &["list", "-j"], None);
    let rows: serde_json::Value = serde_json::from_str(&list.stdout).unwrap();
    assert!(
        rows.as_array()
            .unwrap()
            .iter()
            .any(|r| r["acronym"] == "MVP")
    );

    let show = run(&sock, &["show", "MVP", "-j"], None);
    let shown: serde_json::Value = serde_json::from_str(&show.stdout).unwrap();
    assert!(
        shown
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["expansion"] == "Minimum Viable Product")
    );

    // `list <filter>` folds in the old `search`.
    let search = run(&sock, &["list", "viable", "-j"], None);
    let found: serde_json::Value = serde_json::from_str(&search.stdout).unwrap();
    assert!(
        found
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["acronym"] == "MVP")
    );

    // An added acronym now expands and is no longer a candidate.
    let analyze = run(&sock, &["-j"], Some("ship the MVP"));
    let a: Vec<serde_json::Value> = serde_json::from_str(&analyze.stdout).unwrap();
    assert!(
        a.iter()
            .any(|f| f["kind"] == "expansion" && f["acronym"] == "MVP")
    );
    assert!(
        !a.iter()
            .any(|f| f["kind"] == "candidate" && f["acronym"] == "MVP")
    );

    let rm = run(&sock, &["rm", "MVP", "-j"], None);
    let removed: serde_json::Value = serde_json::from_str(&rm.stdout).unwrap();
    assert_eq!(removed["removed"], 1);
}

#[test]
fn rm_disambiguates_among_multiple_variants() {
    let sock = scratch_socket("rmvariants");
    run(&sock, &["add", "MVP", "Minimum Viable Product"], None);
    run(&sock, &["add", "MVP", "Most Valuable Player"], None);

    // Bare rm refuses when several variants exist.
    assert!(!run(&sock, &["rm", "MVP"], None).success);

    // A substring picks exactly one.
    let one = run(&sock, &["rm", "MVP", "valuable", "-j"], None);
    let v: serde_json::Value = serde_json::from_str(&one.stdout).unwrap();
    assert_eq!(v["removed"], 1);

    // With one left, bare rm removes it.
    let rest = run(&sock, &["rm", "MVP", "-j"], None);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&rest.stdout).unwrap()["removed"],
        1
    );
}

#[test]
fn rm_all_removes_every_variant() {
    let sock = scratch_socket("rmall");
    run(&sock, &["add", "PT", "Physical Therapy"], None);
    run(&sock, &["add", "PT", "Part Time"], None);
    let out = run(&sock, &["rm", "PT", "--all", "-j"], None);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&out.stdout).unwrap()["removed"],
        2
    );
}

#[test]
fn candidates_command_lists_undefined_acronyms_with_counts() {
    let sock = scratch_socket("cands");
    // Analysis surfaces and records MVP as a candidate.
    run(&sock, &["-j"], Some("ship the MVP"));
    let out = run(&sock, &["candidates", "-j"], None);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert!(
        v.as_array()
            .unwrap()
            .iter()
            .any(|c| c["acronym"] == "MVP" && c["count"].as_i64().unwrap() >= 1)
    );
}

#[test]
fn suggest_surfaces_mined_expansions_with_confidence() {
    let sock = scratch_socket("suggest");
    // MVP is undefined; the phrase is mentioned in the same text (no parens).
    run(
        &sock,
        &[],
        Some("the MVP plan: minimum viable product, then iterate"),
    );
    let out = run(&sock, &["suggest", "MVP", "-j"], None);
    assert!(out.success, "stderr: {}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert!(
        v.as_array()
            .unwrap()
            .iter()
            .any(|s| s["expansion"] == "minimum viable product"
                && s["confidence"].as_f64().unwrap() > 0.0),
        "no suggestion mined: {}",
        out.stdout
    );
}

#[test]
fn define_adds_multiple_expansions_at_once() {
    let sock = scratch_socket("define");
    let out = run(
        &sock,
        &[
            "define",
            "MVP",
            "Minimum Viable Product",
            "Most Valuable Player",
            "-j",
        ],
        None,
    );
    assert!(out.success, "stderr: {}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["added"].as_array().unwrap().len(), 2);
    // Both are now in the dictionary.
    let show = run(&sock, &["show", "MVP", "-j"], None);
    let s: serde_json::Value = serde_json::from_str(&show.stdout).unwrap();
    assert_eq!(s.as_array().unwrap().len(), 2);
}

#[test]
fn prune_dedups_prefix_variants() {
    let sock = scratch_socket("prune");
    // Two near-duplicate speculative expansions accrue for MVP.
    run(&sock, &[], Some("the MVP is a min viable product"));
    run(&sock, &[], Some("our MVP, a minimum viable product"));

    let p = run(&sock, &["prune", "-j"], None);
    let pv: serde_json::Value = serde_json::from_str(&p.stdout).unwrap();
    assert!(
        pv["merged"].as_i64().unwrap() >= 1,
        "nothing merged: {}",
        p.stdout
    );

    // Deduped to the fuller canonical form.
    let after = run(
        &sock,
        &["suggest", "MVP", "--min-confidence", "0", "-j"],
        None,
    );
    let av: serde_json::Value = serde_json::from_str(&after.stdout).unwrap();
    assert!(
        av.as_array()
            .unwrap()
            .iter()
            .any(|s| s["expansion"] == "minimum viable product")
    );
}

#[test]
fn suggest_respects_min_confidence() {
    let sock = scratch_socket("suggestmin");
    run(&sock, &[], Some("the MVP plan: minimum viable product"));
    // An impossible threshold hides everything.
    let hidden = run(
        &sock,
        &["suggest", "MVP", "--min-confidence", "1.1", "-j"],
        None,
    );
    let v: serde_json::Value = serde_json::from_str(&hidden.stdout).unwrap();
    assert!(v.as_array().unwrap().is_empty());
}

#[test]
fn add_accepts_multiple_expansions() {
    let sock = scratch_socket("addmulti");
    let out = run(
        &sock,
        &["add", "PT", "Physical Therapy", "Part Time", "-j"],
        None,
    );
    assert!(out.success, "stderr: {}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["added"].as_array().unwrap().len(), 2);
    let show = run(&sock, &["show", "PT", "-j"], None);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&show.stdout)
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn quiet_suppresses_output_but_still_works() {
    let sock = scratch_socket("quiet");
    // Analysis with -q prints nothing.
    let a = run(&sock, &["-q"], Some("Check the OKR board."));
    assert!(a.success && a.stdout.is_empty(), "stdout: {}", a.stdout);
    // A command with -q prints nothing but still mutates.
    let add = run(&sock, &["add", "XY", "Example Co", "-q"], None);
    assert!(add.success && add.stdout.is_empty());
    let show = run(&sock, &["show", "XY", "-j"], None);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&show.stdout)
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn suggest_limit_caps_per_acronym() {
    let sock = scratch_socket("limit");
    run(&sock, &[], Some("the MVP is a minimum viable product"));
    run(&sock, &[], Some("MVP, a most valuable player"));
    let out = run(
        &sock,
        &[
            "suggest",
            "MVP",
            "--min-confidence",
            "0",
            "--limit",
            "1",
            "-j",
        ],
        None,
    );
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v.as_array().unwrap().len(), 1);
}

#[test]
fn list_marks_verified_source() {
    let sock = scratch_socket("verified");
    run(&sock, &["add", "ZZ", "Zig Zag"], None);
    let out = run(&sock, &["list", "-j"], None);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert!(
        v.as_array()
            .unwrap()
            .iter()
            .any(|r| r["acronym"] == "ZZ" && r["source"] == "user" && r["verified"] == true)
    );
}

#[test]
fn expansion_findings_carry_a_confidence() {
    let sock = scratch_socket("confidence");
    let out = run(&sock, &["-j"], Some("Check the OKR board."));
    let v: Vec<serde_json::Value> = serde_json::from_str(&out.stdout).unwrap();
    let m = v
        .iter()
        .find(|f| f["kind"] == "expansion" && f["acronym"] == "OKR")
        .expect("OKR expansion finding");
    // A single trust score is exposed; provenance/validity stays internal.
    assert!(m["confidence"].as_f64().is_some());
    assert!(m.get("validity").is_none());
}

#[test]
fn punctuated_acronym_is_a_candidate_and_mines() {
    let sock = scratch_socket("pbj");
    let a = run(&sock, &["-j"], Some("a PB&J is peanut butter and jelly"));
    let v: Vec<serde_json::Value> = serde_json::from_str(&a.stdout).unwrap();
    assert!(
        v.iter()
            .any(|f| f["kind"] == "candidate" && f["acronym"] == "PB&J")
    );

    let s = run(
        &sock,
        &["suggest", "PB&J", "--min-confidence", "0", "-j"],
        None,
    );
    let sv: serde_json::Value = serde_json::from_str(&s.stdout).unwrap();
    assert!(
        sv.as_array()
            .unwrap()
            .iter()
            .any(|x| x["expansion"] == "peanut butter and jelly")
    );
}

#[test]
fn add_without_expansion_declares_an_acronym() {
    let sock = scratch_socket("declare");
    // `add ACR` with no expansion declares it (was the `watch` command).
    let out = run(&sock, &["add", "MVP", "-j"], None);
    assert!(out.success, "stderr: {}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["status"], "watching");
    // It shows up as a declared candidate.
    let c = run(&sock, &["candidates", "-j"], None);
    let cv: serde_json::Value = serde_json::from_str(&c.stdout).unwrap();
    assert!(
        cv.as_array()
            .unwrap()
            .iter()
            .any(|r| r["acronym"] == "MVP" && r["source"] == "declared" && r["watching"] == true)
    );
}

#[test]
fn auto_consolidation_runs_after_a_write_and_prunes_noise() {
    let sock = scratch_socket("autogc");
    // Consolidate due on every write (interval 0), no grace, so the seen-once
    // noise candidate is cleaned up by the pass that fires after this analysis.
    let env = [("AE_CONSOLIDATE_SECS", "0"), ("AE_PRUNE_GRACE_SECS", "0")];
    let a = run_with_env(&sock, &["-j"], Some("the ZZQ widget"), &env);
    let v: Vec<serde_json::Value> = serde_json::from_str(&a.stdout).unwrap();
    assert!(
        v.iter()
            .any(|f| f["kind"] == "candidate" && f["acronym"] == "ZZQ")
    );

    // The candidate is gone — consolidation pruned it (read-only command won't refire).
    let c = run_with_env(&sock, &["candidates", "-j"], None, &env);
    let cv: serde_json::Value = serde_json::from_str(&c.stdout).unwrap();
    assert!(cv.as_array().unwrap().iter().all(|r| r["acronym"] != "ZZQ"));
}

#[test]
fn plain_text_reports_no_findings() {
    let sock = scratch_socket("plain");
    let out = run(&sock, &[], Some("just an ordinary lowercase sentence"));
    assert!(out.success);
    assert!(out.stdout.contains("No acronyms"));
}

#[test]
fn empty_piped_input_finds_nothing() {
    let sock = scratch_socket("empty");
    // Empty stream: no lines to analyze — benign, not an error.
    let out = run_piped(&sock, &[], "");
    assert!(out.success);
    assert!(out.stdout.contains("No acronyms"));
}

#[test]
fn logs_go_to_stderr_not_stdout() {
    let sock = scratch_socket("streams");
    // --verbose forces telemetry; stdout must still be parseable JSON.
    let out = run(&sock, &["--verbose", "-j"], Some("Check the API docs."));
    assert!(out.success, "stderr: {}", out.stderr);
    serde_json::from_str::<serde_json::Value>(&out.stdout).expect("stdout stayed pristine JSON");
}
