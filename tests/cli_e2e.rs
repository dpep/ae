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

#[test]
fn all_caps_input_is_not_flagged_as_acronyms() {
    let sock = scratch_socket("allcaps");
    // A shouted, all-caps line: every word is acronym-shaped, but with no
    // lowercase contrast none should be flagged as a candidate.
    let out = run(&sock, &["-j"], Some("SHIP THE NEW MVP TODAY"));
    assert!(out.success, "stderr: {}", out.stderr);
    let v: Vec<serde_json::Value> = serde_json::from_str(&out.stdout).unwrap();
    assert!(
        !v.iter().any(|f| f["kind"] == "candidate"),
        "all-caps flooded candidates: {}",
        out.stdout
    );
}

#[test]
fn ignore_makes_an_acronym_inert_and_unignore_restores_it() {
    let sock = scratch_socket("ignore");

    // Baseline: OKR (seeded) expands.
    let before = run(&sock, &["-j"], Some("check the OKR"));
    let bv: Vec<serde_json::Value> = serde_json::from_str(&before.stdout).unwrap();
    assert!(
        bv.iter()
            .any(|f| f["kind"] == "expansion" && f["acronym"] == "OKR")
    );

    // Mute it (machine status), then it no longer surfaces at all.
    let ig = run(&sock, &["ignore", "OKR", "-j"], None);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&ig.stdout).unwrap()["status"],
        "ignored"
    );
    let muted = run(&sock, &["-j"], Some("check the OKR"));
    let mv: Vec<serde_json::Value> = serde_json::from_str(&muted.stdout).unwrap();
    assert!(
        !mv.iter().any(|f| f["acronym"] == "OKR"),
        "OKR still surfaced while ignored: {}",
        muted.stdout
    );

    // It's listed by `ignore` with no argument...
    let list = run(&sock, &["ignore", "-j"], None);
    let lv: Vec<serde_json::Value> = serde_json::from_str(&list.stdout).unwrap();
    assert!(lv.iter().any(|r| r["acronym"] == "OKR"));

    // ...but the entry was not deleted — `show` still has the expansion.
    let show = run(&sock, &["show", "OKR", "-j"], None);
    let sv: Vec<serde_json::Value> = serde_json::from_str(&show.stdout).unwrap();
    assert!(
        sv.iter()
            .any(|r| r["expansion"] == "Objectives and Key Results")
    );

    // Un-mute restores expansion.
    let un = run(&sock, &["unignore", "OKR", "-j"], None);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&un.stdout).unwrap()["status"],
        "unignored"
    );
    let after = run(&sock, &["-j"], Some("check the OKR"));
    let av: Vec<serde_json::Value> = serde_json::from_str(&after.stdout).unwrap();
    assert!(
        av.iter()
            .any(|f| f["kind"] == "expansion" && f["acronym"] == "OKR")
    );
}

#[test]
fn rm_all_wipes_with_a_backup_and_restore_recovers() {
    let sock = scratch_socket("wipeall");
    assert!(run(&sock, &["add", "MVP", "Minimum Viable Product"], None).success);

    // `rm --all` (no acronym) wipes everything, reporting a backup path.
    let wiped = run(&sock, &["rm", "--all", "-j"], None);
    assert!(wiped.success, "stderr: {}", wiped.stderr);
    let wv: serde_json::Value = serde_json::from_str(&wiped.stdout).unwrap();
    assert_eq!(wv["status"], "cleared");
    let backup = wv["backup"].as_str().unwrap().to_string();
    assert!(
        std::path::Path::new(&backup).is_file(),
        "backup file missing: {backup}"
    );

    // MVP is gone; the built-in defaults re-seed (factory reset).
    let after = run(&sock, &["list", "-j"], None);
    let av: Vec<serde_json::Value> = serde_json::from_str(&after.stdout).unwrap();
    assert!(!av.iter().any(|r| r["acronym"] == "MVP"));
    assert!(av.iter().any(|r| r["acronym"] == "OKR"));

    // Restoring the backup brings MVP back.
    let restored = run(&sock, &["rm", "--restore", backup.as_str(), "-j"], None);
    assert!(restored.success, "stderr: {}", restored.stderr);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&restored.stdout).unwrap()["status"],
        "restored"
    );
    let relist = run(&sock, &["list", "-j"], None);
    let rv: Vec<serde_json::Value> = serde_json::from_str(&relist.stdout).unwrap();
    assert!(rv.iter().any(|r| r["acronym"] == "MVP"));
    let _ = std::fs::remove_file(&backup);
}

#[test]
fn rm_with_no_acronym_and_no_all_is_an_error() {
    let sock = scratch_socket("rmnoargs");
    assert!(!run(&sock, &["rm"], None).success);
}

#[test]
fn adding_an_expansion_unmutes_an_ignored_acronym() {
    let sock = scratch_socket("addunmute");

    // Mute OKR, confirm it's inert.
    assert!(run(&sock, &["ignore", "OKR"], None).success);
    let muted = run(&sock, &["-j"], Some("check the OKR"));
    let mv: Vec<serde_json::Value> = serde_json::from_str(&muted.stdout).unwrap();
    assert!(!mv.iter().any(|f| f["acronym"] == "OKR"));

    // Explicitly defining it reactivates it — no separate `unignore` needed.
    assert!(run(&sock, &["add", "OKR", "Objectives and Key Results"], None).success);
    assert!(
        run(&sock, &["ignore", "-j"], None)
            .stdout
            .trim()
            .ends_with("[]")
    );
    let after = run(&sock, &["-j"], Some("check the OKR"));
    let av: Vec<serde_json::Value> = serde_json::from_str(&after.stdout).unwrap();
    assert!(
        av.iter()
            .any(|f| f["kind"] == "expansion" && f["acronym"] == "OKR")
    );
}

// --- rm: exact-match selection, --unverified, case-insensitive add ---

#[test]
fn rm_exact_expansion_beats_a_substring_sibling() {
    let sock = scratch_socket("rmexact");
    run(&sock, &["add", "PP", "Partner Platform"], None);
    run(&sock, &["add", "PP", "Agent and Partner Platform"], None);
    let out = run(&sock, &["rm", "PP", "Partner Platform", "-j"], None);
    assert!(out.success, "{}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["removed"], 1);
    let left = run(&sock, &["show", "PP", "-j"], None);
    let rows: Vec<serde_json::Value> = serde_json::from_str(&left.stdout).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["expansion"], "Agent and Partner Platform");
}

/// Expansions of `acronym` currently shown by `ae show`.
fn shown(sock: &std::path::Path, acronym: &str) -> Vec<String> {
    let out = run(sock, &["show", acronym, "-j"], None);
    let rows: Vec<serde_json::Value> = serde_json::from_str(&out.stdout).unwrap();
    rows.iter()
        .map(|r| r["expansion"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn rm_unverified_keeps_user_rows_of_one_acronym() {
    let sock = scratch_socket("rmunverified");
    run(&sock, &[], Some("The ZQ (Zebra Queue) is deep.")); // learned inline
    run(&sock, &["add", "ZQ", "Zone Quota"], None);
    run(&sock, &[], Some("The WQ (Widget Quorum) met.")); // another acronym

    let out = run(&sock, &["rm", "ZQ", "--unverified", "-j"], None);
    assert!(out.success, "{}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["status"], "removed");
    assert_eq!(v["acronym"], "ZQ");
    assert_eq!(v["removed"], 1);
    assert!(v["backup"].is_null());

    assert_eq!(shown(&sock, "ZQ"), vec!["Zone Quota"]);
    assert_eq!(shown(&sock, "WQ"), vec!["Widget Quorum"]); // untouched
}

#[test]
fn rm_unverified_without_an_acronym_cleans_everything_with_a_backup() {
    let sock = scratch_socket("rmunverifiedall");
    run(&sock, &[], Some("The ZQ (Zebra Queue) is deep."));
    run(&sock, &[], Some("The WQ (Widget Quorum) met."));
    run(&sock, &["add", "ZQ", "Zone Quota"], None);

    let out = run(&sock, &["rm", "--unverified", "-j"], None);
    assert!(out.success, "{}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["status"], "removed");
    assert!(v["acronym"].is_null());
    assert_eq!(v["removed"], 2);
    let backup = v["backup"].as_str().expect("a backup path");
    assert!(std::path::Path::new(backup).is_file());

    assert_eq!(shown(&sock, "ZQ"), vec!["Zone Quota"]);
    assert!(shown(&sock, "WQ").is_empty());
    assert!(!shown(&sock, "KPI").is_empty()); // built-in defaults are user rows

    // The backup brings the inline rows back.
    assert!(run(&sock, &["rm", "--restore", backup], None).success);
    assert_eq!(shown(&sock, "WQ"), vec!["Widget Quorum"]);
}

#[test]
fn rm_unverified_conflicts_with_picking_an_expansion() {
    let sock = scratch_socket("rmunverifiedconflict");
    assert!(!run(&sock, &["rm", "ZQ", "Zebra", "--unverified"], None).success);
    assert!(!run(&sock, &["rm", "ZQ", "--all", "--unverified"], None).success);
}

#[test]
fn add_folds_an_expansion_differing_only_in_case() {
    let sock = scratch_socket("addcase");
    run(&sock, &["add", "BK", "buildkite"], None);
    run(&sock, &["add", "BK", "Buildkite"], None);
    assert_eq!(shown(&sock, "BK").len(), 1);
}

// --- prune clears non-acronym candidates; `ae stats` ---

/// Open the test's DB directly, to plant rows the CLI can no longer create
/// (candidates recorded before the noise rule existed).
fn store_of(sock: &std::path::Path) -> acronym_engine::store::Store {
    acronym_engine::store::Store::open(&sock.with_extension("db")).unwrap()
}

fn candidate_names(sock: &std::path::Path) -> Vec<String> {
    let out = run(sock, &["candidates", "-j"], None);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    v.as_array()
        .unwrap()
        .iter()
        .map(|c| c["acronym"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn prune_clears_recorded_non_acronyms_but_not_declared_ones() {
    let sock = scratch_socket("prunenoise");
    {
        // TODO/V1/INFO are noise with or without a system wordlist.
        let s = store_of(&sock);
        for _ in 0..3 {
            s.record_candidate("TODO").unwrap();
            s.record_candidate("V1").unwrap();
            s.record_candidate("ZQX").unwrap();
        }
        for _ in 0..2 {
            s.record_potential("TODO", "the other day only", 0.9)
                .unwrap();
        }
        s.declare_acronym("INFO").unwrap();
    }

    let out = run(&sock, &["prune", "-j"], None);
    assert!(out.success, "{}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["non_acronyms_removed"], 2);
    assert_eq!(v["non_acronym_expansions_removed"], 1);

    let mut names = candidate_names(&sock);
    names.sort();
    assert_eq!(names, ["INFO", "ZQX"]);

    let human = run(&sock, &["prune"], None);
    assert!(human.stdout.contains("non-acronym"), "{}", human.stdout);
}

/// Six built-ins, one user entry, an aligned and an unaligned inline entry, a
/// mined row, and two candidates (one watched).
fn stats_fixture(label: &str) -> PathBuf {
    let sock = scratch_socket(label);
    let s = store_of(&sock);
    s.seed_defaults().unwrap();
    s.add_entry("ZQ", "Zone Quota", "user").unwrap();
    s.add_entry("WQ", "Widget Quorum", "inline").unwrap();
    s.add_entry("XY", "Red Balloon", "inline").unwrap();
    s.record_potential("ZQX", "zebra queue xray", 0.9).unwrap();
    for _ in 0..3 {
        s.record_candidate("ZQX").unwrap();
    }
    s.record_candidate("QQA").unwrap();
    sock
}

#[test]
fn stats_reports_counts_and_shares_built_from_them() {
    let sock = stats_fixture("stats");
    let out = run(&sock, &["stats", "-j"], None);
    assert!(out.success, "{}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let want = serde_json::json!({
        "confirmed": 9, "acronyms": 9,
        "verified": 7, "builtin": 6, "verified_share": 0.78,
        "inline": 2, "aligned": 1, "aligned_share": 0.5,
        "mined": 1, "candidates": 2, "watched": 1,
    });
    assert_eq!(v, want);

    let nd = run(&sock, &["stats", "-J"], None);
    assert_eq!(nd.stdout.lines().count(), 1);
    let line: serde_json::Value = serde_json::from_str(&nd.stdout).unwrap();
    assert_eq!(line, want);

    let human = run(&sock, &["stats"], None);
    assert!(human.stdout.contains("78%"), "{}", human.stdout);
    assert!(human.stdout.contains("50%"), "{}", human.stdout);
}

#[test]
fn stats_on_a_fresh_dictionary_has_no_aligned_share() {
    let sock = scratch_socket("statsfresh");
    let out = run(&sock, &["stats", "-j"], None);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let builtins = acronym_engine::store::DEFAULT_DICTIONARY.len();
    assert_eq!(v["confirmed"], builtins);
    assert_eq!(v["builtin"], builtins);
    assert_eq!(v["verified_share"], 1.0);
    assert!(v["aligned_share"].is_null());
}

// --- cleanup safety: per-database backups, case folding, prune to current rules ---

/// This test DB's backups: `<db dir>/backups/<db stem>-*.db`.
fn backups_of(sock: &std::path::Path) -> Vec<PathBuf> {
    let db = sock.with_extension("db");
    let stem = db.file_stem().unwrap().to_str().unwrap().to_string();
    let Ok(dir) = std::fs::read_dir(db.parent().unwrap().join("backups")) else {
        return Vec::new();
    };
    dir.filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&format!("{stem}-")))
        })
        .collect()
}

#[test]
fn a_bare_restore_never_picks_another_databases_backup() {
    let a = scratch_socket("restoremine-a");
    let b = scratch_socket("restoremine-b");
    run(&a, &[], Some("The ZQ (Zebra Queue) is deep."));
    run(&b, &[], Some("The WQ (Widget Quorum) met."));

    assert!(run(&a, &["rm", "--unverified"], None).success);
    assert!(run(&b, &["rm", "--unverified"], None).success); // newer, other DB
    let out = run(&a, &["rm", "--restore", "-j"], None);
    assert!(out.success, "{}", out.stderr);

    assert_eq!(shown(&a, "ZQ"), vec!["Zebra Queue"]);
    assert!(shown(&a, "WQ").is_empty(), "restored B's contents into A");
}

#[test]
fn backups_live_beside_the_database_and_are_private() {
    use std::os::unix::fs::PermissionsExt;
    let sock = scratch_socket("backupbeside");
    run(&sock, &[], Some("The ZQ (Zebra Queue) is deep."));
    let out = run(&sock, &["rm", "--unverified", "-j"], None);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let backup = PathBuf::from(v["backup"].as_str().expect("a backup path"));

    assert_eq!(backups_of(&sock), vec![backup.clone()]);
    let mode = std::fs::metadata(&backup).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o600, "backup is readable by others");
}

#[test]
fn a_bulk_remove_with_nothing_to_remove_takes_no_backup() {
    let sock = scratch_socket("backupnoop");
    let out = run(&sock, &["rm", "--unverified", "-j"], None);
    assert!(out.success, "{}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["removed"], 0);
    assert!(v["backup"].is_null());
    assert!(backups_of(&sock).is_empty());
}

#[test]
fn a_bare_restore_with_no_backups_for_this_database_fails() {
    let sock = scratch_socket("restorenone");
    assert!(!run(&sock, &["rm", "--restore"], None).success);
}

#[test]
fn prune_brings_legacy_confirmed_rows_up_to_current_rules_with_a_backup() {
    let sock = scratch_socket("prunelegacy");
    store_of(&sock).seed_defaults().unwrap();
    // Rows an older `ae` wrote: an unaligned inline definition, and case-only
    // siblings (add_entry folds those now, so plant them directly).
    let db = rusqlite::Connection::open(sock.with_extension("db")).unwrap();
    db.execute_batch(
        "INSERT INTO acronym_dictionary (acronym, expansion, source) VALUES
           ('XY', 'Red Balloon', 'inline'),
           ('WQ', 'Widget Quorum', 'inline'),
           ('WQ', 'widget quorum', 'inline');",
    )
    .unwrap();
    drop(db);

    let out = run(&sock, &["prune", "-j"], None);
    assert!(out.success, "{}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["unaligned_removed"], 1);
    assert_eq!(v["case_duplicates_merged"], 1);
    let backup = v["backup"].as_str().expect("a backup path");
    assert_eq!(backups_of(&sock), vec![PathBuf::from(backup)]);
    assert!(shown(&sock, "XY").is_empty());
    assert_eq!(shown(&sock, "WQ"), vec!["Widget Quorum"]);

    // Nothing left to bring up to date: no backup.
    let again = run(&sock, &["prune", "-j"], None);
    let v: serde_json::Value = serde_json::from_str(&again.stdout).unwrap();
    assert_eq!(v["unaligned_removed"], 0);
    assert!(v["backup"].is_null());

    let human = run(&sock, &["prune"], None);
    assert!(human.stdout.contains("unaligned"), "{}", human.stdout);
}

/// Run `ae` with its stdout closed before it writes — `ae … | head` once head
/// has had enough. Returns (success, stderr).
fn run_into_closed_pipe(sock: &std::path::Path, args: &[&str], stdin: &str) -> (bool, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_ae"))
        .arg("--socket")
        .arg(sock)
        .arg("--db")
        .arg(sock.with_extension("db"))
        .args(args)
        .env("AE_CONSOLIDATE_SECS", "-1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    // `ae` may stop reading once its output has nowhere to go.
    let _ = child.stdin.take().unwrap().write_all(stdin.as_bytes());
    let out = child.wait_with_output().unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn output_into_a_closed_pipe_exits_quietly() {
    let sock = scratch_socket("epipe");
    {
        // More output than a pipe buffers, so the write fails even if the
        // reader hangs up late.
        let s = store_of(&sock);
        for i in 0..4000 {
            s.declare_acronym(&format!("ZQ{i}X")).unwrap();
        }
    }
    let lines = "Check the KPI and the OKR.\n".repeat(3000);
    for (args, stdin) in [
        (&["candidates"][..], ""),
        (&["candidates", "-j"][..], ""),
        (&["-r"][..], lines.as_str()),
        (&["-r", "-J"][..], lines.as_str()),
    ] {
        let (ok, stderr) = run_into_closed_pipe(&sock, args, stdin);
        assert!(ok, "{args:?} failed: {stderr}");
        assert!(stderr.is_empty(), "{args:?} complained: {stderr}");
    }
}

// --- Precision: matching, alignment, and mining only claim what the text shows ---

/// Acronyms of every expansion finding for `text` (read-only, JSON).
fn expanded_acronyms(sock: &std::path::Path, text: &str) -> Vec<String> {
    let out = run(sock, &["-rj"], Some(text));
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let mut names: Vec<String> = v
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["kind"] == "expansion")
        .map(|f| f["acronym"].as_str().unwrap().to_string())
        .collect();
    names.dedup();
    names
}

#[test]
fn an_ordinary_lowercase_word_does_not_expand_as_an_acronym() {
    let sock = scratch_socket("lowercaseword");
    assert!(run(&sock, &["add", "US", "United States"], None).success);
    assert!(run(&sock, &["add", "IT", "Information Technology"], None).success);
    assert!(expanded_acronyms(&sock, "can you help us fix it").is_empty());
    assert_eq!(
        expanded_acronyms(&sock, "the US office runs IT"),
        ["US", "IT"]
    );
}

#[test]
fn an_inline_definition_learns_only_when_its_initials_spell_the_acronym() {
    let sock = scratch_socket("alignlearn");
    // Camel case: each capital starts a word, so this spells WSON.
    run(
        &sock,
        &["-q"],
        Some("Send WSON (WidgetSoft Object Notation) over."),
    );
    assert_eq!(expanded_acronyms(&sock, "parse the WSON file"), ["WSON"]);

    // A fragment that doesn't spell KPI is reported but leaves no suggestion.
    let out = run(&sock, &["-j"], Some("Ping the Platform Team (KPI) today"));
    assert!(out.stdout.contains("extraction"), "{}", out.stdout);
    let s = run(
        &sock,
        &["suggest", "KPI", "--min-confidence", "0", "-j"],
        None,
    );
    let v: serde_json::Value = serde_json::from_str(&s.stdout).unwrap();
    assert!(v.as_array().unwrap().is_empty(), "{}", s.stdout);
}
