//! End-to-end tests that drive the built `ae` binary with an isolated socket
//! and DB, exercising the no-daemon self-healing fallback path.
//!
//! Input is fed via stdin (the pipe path): when stdin isn't a TTY — which is
//! always the case under the test harness — `ae` consumes it, so we pipe text
//! rather than passing it as an argument.

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

/// Run `ae` with `args`, feeding `stdin` (or an empty closed stdin if `None`).
/// Auto-consolidation is disabled so tests are deterministic.
fn run(socket: &std::path::Path, args: &[&str], stdin: Option<&str>) -> Output {
    run_with_env(socket, args, stdin, &[("AE_CONSOLIDATE_SECS", "-1")])
}

/// Like [`run`], but with explicit env overrides — e.g. forcing GC on.
fn run_with_env(
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
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert!(v["sentence"].is_string());
    let learned = v["extractions"].as_array().unwrap();
    assert!(learned.iter().any(|c| c["acronym"] == "KPI"));
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
    let v: serde_json::Value = serde_json::from_str(&second.stdout).unwrap();
    let expansions = v["expansions"].as_array().unwrap();
    assert!(
        expansions
            .iter()
            .any(|e| e["acronym"] == "ZQ" && e["matches"][0]["expansion"] == "Zebra Queue"),
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
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(v["expansions"][0]["acronym"], "OKR");
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
    let v: serde_json::Value = serde_json::from_str(&first.stdout).unwrap();
    assert!(
        v["expansions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["acronym"] == "OKR")
    );
    assert!(v["extractions"].as_array().unwrap().is_empty());

    // ...and nothing was persisted: a normal later pass still doesn't know ZQ.
    let second = run(&sock, &["-j"], Some("Drain the ZQ."));
    let v2: serde_json::Value = serde_json::from_str(&second.stdout).unwrap();
    assert!(
        !v2["expansions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["acronym"] == "ZQ")
    );
}

#[test]
fn short_json_and_ndjson_flags_select_the_format() {
    let sock = scratch_socket("shortflags");
    // -j → a single pretty JSON object.
    let j = run(&sock, &["-j"], Some("Check the OKR board."));
    serde_json::from_str::<serde_json::Value>(&j.stdout).expect("-j is valid JSON");

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
fn batch_mode_aggregates_per_line_hits_with_positions() {
    let sock = scratch_socket("batch");
    let input = "first line has an OKR\nsecond mentions the API\n";
    let out = run(&sock, &["--batch", "-j"], Some(input));
    assert!(out.success, "stderr: {}", out.stderr);
    let hits: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let arr = hits.as_array().unwrap();
    assert!(
        arr.iter()
            .any(|h| h["acronym"] == "OKR" && h["line"] == 1 && h["col"].as_u64().unwrap() > 0)
    );
    assert!(arr.iter().any(|h| h["acronym"] == "API" && h["line"] == 2));
}

#[test]
fn file_flag_reads_a_file_and_implies_batch() {
    let sock = scratch_socket("file");
    let path = std::env::temp_dir().join(format!("ae-input-{}.txt", std::process::id()));
    std::fs::write(&path, "intro line\nthis row has the OKR\n").unwrap();
    // No stdin, no --batch — the file alone triggers aggregated output.
    let out = run(&sock, &["--file", path.to_str().unwrap(), "-j"], None);
    assert!(out.success, "stderr: {}", out.stderr);
    let hits: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    assert!(
        hits.as_array()
            .unwrap()
            .iter()
            .any(|h| h["acronym"] == "OKR" && h["line"] == 2)
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn batch_human_output_is_grep_style() {
    let sock = scratch_socket("batchhuman");
    let out = run(&sock, &["-b"], Some("line one\nthe OKR is here\n"));
    assert!(out.success, "stderr: {}", out.stderr);
    // line:col: ACR ... — the OKR is on line 2.
    assert!(
        out.stdout
            .lines()
            .any(|l| l.starts_with("2:") && l.contains("OKR"))
    );
}

#[test]
fn unknown_acronyms_are_surfaced() {
    let sock = scratch_socket("unknown");
    let out = run(&sock, &["-j"], Some("hi there MVP"));
    assert!(out.success, "stderr: {}", out.stderr);
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let unknown = v["candidates"].as_array().unwrap();
    assert!(
        unknown.iter().any(|u| u == "MVP"),
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
    let a: serde_json::Value = serde_json::from_str(&analyze.stdout).unwrap();
    assert!(
        a["expansions"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["acronym"] == "MVP")
    );
    assert!(
        !a["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c == "MVP")
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
fn analysis_matches_carry_validity_and_confidence() {
    let sock = scratch_socket("validity");
    let out = run(&sock, &["-j"], Some("Check the OKR board."));
    let v: serde_json::Value = serde_json::from_str(&out.stdout).unwrap();
    let m = &v["expansions"][0]["matches"][0];
    // OKR is a curated (user) default → fully valid.
    assert_eq!(m["validity"], 1.0);
    assert!(m["confidence"].as_f64().is_some());
}

#[test]
fn punctuated_acronym_is_a_candidate_and_mines() {
    let sock = scratch_socket("pbj");
    let a = run(&sock, &["-j"], Some("a PB&J is peanut butter and jelly"));
    let v: serde_json::Value = serde_json::from_str(&a.stdout).unwrap();
    assert!(
        v["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c == "PB&J")
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
    let v: serde_json::Value = serde_json::from_str(&a.stdout).unwrap();
    assert!(
        v["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c == "ZZQ")
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
fn empty_input_is_an_error() {
    let sock = scratch_socket("empty");
    let out = run(&sock, &[], Some(""));
    assert!(!out.success);
    assert!(out.stderr.contains("no input"));
}

#[test]
fn logs_go_to_stderr_not_stdout() {
    let sock = scratch_socket("streams");
    // --verbose forces telemetry; stdout must still be parseable JSON.
    let out = run(&sock, &["--verbose", "-j"], Some("Check the API docs."));
    assert!(out.success, "stderr: {}", out.stderr);
    serde_json::from_str::<serde_json::Value>(&out.stdout).expect("stdout stayed pristine JSON");
}
