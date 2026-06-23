---
name: ROADMAP
description: Implementation progress, deviations from the spec, feature requests, and known bugs for ae. The living tracker — update it in the same change that moves the work.
---

# ae roadmap

Phased plan mirroring [SPEC.md](SPEC.md). Each milestone is independently useful
and ends in something you can actually run. This file is the living tracker:
when you land a change, tick its box and note any deviation here in the same
commit.

## Status legend

- [x] done and tested
- [~] partial / stubbed (see note)
- [ ] not started

## Milestone 1 — CLI, stream isolation, piping

- [x] `Cli` struct + `Format {human,json,ndjson}` (clap derive)
- [x] input resolution: positional arg wins; else non-TTY stdin → read to string;
      else (TTY, no arg) help/error
- [x] `-d/--daemon` with input ensures a warm daemon, then serves the work through
      it (≈ `ae -d && ae <text>`, one process, prints only the analysis); bare
      `ae -d` just starts it
- [x] `env_logger` to stderr; stdout reserved for data
- [x] `--verbose` raises log level
- [x] unit + e2e tests for input resolution and format flags

## Milestone 2 — IPC socket multiplexing & self-healing

- [x] `flock` on lock file → Leader vs Follower decision
- [x] Leader: UDS listener, one analysis per connection, framed JSON
- [x] Follower: forward text, read JSON reply, render
- [x] `--daemon` detaches a background leader; `--stop` halts it
- [x] lazy janitor: idle-timeout removes the socket and exits
- [x] tests: leader/follower round-trip, idle shutdown, stale-socket recovery

## Milestone 3 — SQLite storage + 64-d MRL compression

- [x] `acronym_dictionary` table + unique `(acronym, expansion)` index, WAL
- [x] `compress_matryoshka_vector`: truncate-64 + L2 normalize + zero guard
- [x] cosine similarity over normalized vectors
- [x] context-embedding store keyed by acronym (plain table — see deviation)
- [x] tests: truncation length, unit-norm output, zero-vector guard, retrieval

## Milestone 4 — Trie + local model evaluation

- [x] thread-safe `SharedTrie` (`RwLock<TrieNode>`); insert / contains / collect
- [x] dual-pipeline payload types (`MatchCandidate` … `AnalysisPayload`)
- [x] `Embedder` trait; deterministic `HashEmbedder` fallback
- [x] real ONNX embedder (`all-MiniLM-L6-v2`, int8-quantized) via ONNX Runtime —
      tokenize → mean-pool → MRL-compress; statically linked
- [x] model fetched on first use from the HuggingFace Hub into the shared
      `~/.cache/huggingface/hub` (via `hf-hub`), never committed and never
      fetched at build time; resolution is `$AE_MODEL_DIR` → Hub → hash fallback
- [x] `--model <path|name>` override; resolution order, named search dirs
- [x] tests: trie insert/scan, mean-pool, model resolution, real-model semantic
      similarity (gated on model presence)

## Milestone 5 — rule-based learning + fallback routing

- [x] Pattern Alpha / Beta extraction (regex), confidence scoring, dedup
- [x] unified `AnalysisPayload` assembly (expansions + learned candidates)
- [x] in-process evaluation engine (Trie + dictionary + embedder)
- [x] fallback routing: UDS client first, in-process engine on connect failure
- [x] learned candidates persisted back to the dictionary
- [x] tests: both patterns, no false positives, fallback path, persistence

## Deviations from the spec (with rationale)

1. **Model is `all-MiniLM-L6-v2`, not `nomic-embed-text-v1.5`.** The spec names
   nomic (768-d, MRL-trained) but also says "384-dimensional" — a contradiction.
   We started on nomic (worked end-to-end) then switched to all-MiniLM-L6-v2:
   its int8 export is ~22 MB vs nomic's ~130 MB, same BERT inference path
   (input_ids/attention_mask/token_type_ids → mean-pool), 384-d native. It isn't
   MRL-trained, so 384→64 truncation loses a bit more than nomic's would; fine
   for acronym-context disambiguation, and `MRL_DIMS` can rise to 128 to recover
   most of it. The `Embedder` trait makes the model swappable; `--model`
   overrides at runtime. `HashEmbedder` remains the deterministic offline/test
   fallback.

2. **"Quantize and shrink" = fetch the upstream int8 export.** Running a real
   quantizer ourselves needs Python's onnxruntime tooling, which we won't take
   on as a dependency. Fetching the already-quantized artifact gives the same
   size win without it.

3. **Vectors live in a plain `acronym_contexts` table, not a `vec0` virtual
   table.** `sqlite-vec`'s `vec0` needs a loadable extension that `rusqlite`'s
   bundled SQLite doesn't carry. At this corpus size cosine in Rust over the
   candidate set is simpler and fast. Swapping in `vec0` later is a storage-layer
   change behind the same API.

4. **`--stop` and the janitor.** `--stop` connects and sends a shutdown frame;
   the janitor is an idle-timeout watchdog thread. The spec's "stdio
   disconnect" trigger is approximated by an idle-connection timeout, which is
   the portable, testable equivalent.

## CLI ergonomics (machine-friendly surface)

- [x] `-j/--json` + `-J/--ndjson` everywhere; `--format` removed in favor of them
- [x] all commands emit structured status (`--daemon`/`--stop` → `{"status":…}`)
- [x] `--read-only` (`-r`) — expand without learning/persisting
- [x] `--batch` (`-b`) — line-by-line aggregation with `line:col` hits
- [x] `--file` (`-f`) — read a file, implies batch
- [x] bare invocation (no input, interactive) prints `--help`

## Evaluation

- [x] three output buckets: **expansions** (known), **extractions** (defined
      inline), **candidates** (acronym-shaped but unresolved). Field/kind names:
      `expansions`/`expansion`, `extractions`/`extraction`,
      `candidates`/`candidate`
- [x] candidate detection — acronym-shaped tokens (`[A-Z][A-Z0-9]{1,5}`) that are
      neither expanded nor extracted are surfaced in `payload.candidates`

## Dictionary management

- [x] `ae list` / `show <ACR>` / `search <QUERY>` / `add <ACR> <EXP>` /
      `candidates` subcommands (no flags; `-j/-J` honored). Operate on the
      `--db` store directly. Note: a running daemon needs `--stop` to pick up
      manual edits (its in-memory trie is hydrated at start)
- [x] `ae rm <ACR> [substring] [--all]` — removes the only expansion, or one
      picked by substring; refuses (and lists) when ambiguous; `--all` removes all
- [x] candidate tracking — every analysis (not read-only) records undefined
      acronym-shaped tokens with occurrence counts (`candidate_acronyms` table);
      defining an acronym clears it. Surfaced via `ae candidates`
- [x] speculative expansion mining — when a candidate appears in text, scan the
      same text for consecutive word-sequences whose initials spell it (no
      parens) and count recurrences (`potential_expansions`); `ae suggest [ACR]`
      ranks them with confidence (share of the acronym's sightings). Defining an
      acronym clears its suggestions

- [x] filler-tolerant mining — match the acronym as a subsequence over words,
      skipping a closed set of fillers (OKR = Objectives *and* Key Results) yet
      consuming one when it supplies a letter (POC = Point *of* Contact);
      content words must contribute (precision guard)
- [x] cross-text mining — watched candidates (len ≥ 3) are mined from later
      inputs that never mention them, so definitions accrue over time
- [x] context vectors in confidence — per-candidate running-mean context
      embedding (`candidate_contexts`); each mined phrase records the coherence
      of its context (`coh_sum`); `confidence` blends recurrence share with mean
      coherence, damped for sparse counts

- [x] `ae define <ACR>` — add given expansions (multi), or pick interactively
      from suggestions via `fzf` (`--multi`) / numbered-prompt fallback
- [x] prefix-normalization dedup — `ae prune` merges prefix-compatible
      expansions ("min" ≈ "minimum", ≥3-char guard) into the fullest form,
      summing counts/coherence
- [x] `--min-confidence` (default 0.15) on `suggest`; `ae prune` drops below it
      and removes seen-once noise candidates

## CLI consolidation

- [x] `list [FILTER]` folds in `search` (substring of acronym or expansion)
- [x] `add <ACR> <EXP>...` takes multiple expansions; `define` is the
      interactive promote (pick from suggestions) — overlap removed
- [x] `-q/--quiet` (global) suppresses normal stdout (analysis + commands)
- [x] `suggest` — higher default floor (0.30 vs prune's 0.15), `--min-confidence`
      override, `-l/--limit N` per acronym; `define` shows *all* (diagnostic)

## Unified confidence model (done)

Known and speculative expansions are now **one table** (`acronym_dictionary`),
one row per `(acronym, expansion)`, differentiated by `source` on a validity
continuum: `user` (verified by a human) > `inline` (defined in the text) >
`mined` (speculative). Re-adding only *upgrades* the source; a mined phrase that
gets confirmed becomes verified in place. `candidate_acronyms` stays separate —
it's the per-*acronym* "is-this-an-acronym" signal, a different entity.

The three scores are explicit:
1. **is-acronym** — candidate detection (shape) + `candidate_acronyms` frequency.
2. **expansion-validity** — `source_validity()`: user 1.0, inline 0.9, mined =
   the recurrence+coherence confidence. Surfaced as `validity` on a match and
   `verified`/`source` in `list`/`show`.
3. **contextual-likelihood** — `Engine::contextual()`: cosine of the sentence
   against the expansion's recorded contexts. Surfaced as `confidence`.

Views over the one table: `list`/`show` = confirmed (`user`/`inline`); `suggest`
= `mined`; the trie hydrates from confirmed only (a mined-only acronym stays a
candidate).

### Acronym shapes & spelling

- [x] punctuated acronyms — `PB&J`, `R&D`, `U.S.A` detected + mined (the `&`/`.`
      maps to a skipped filler word); **maximal munch** — a longer match wins
      over its parts (`PB&J` beats `PB`) in detection and expansion
- [x] fuzzy dedup at `prune` — merge expansions within a small edit distance
      (Levenshtein), alongside the prefix merge
- [x] dictionary spell-correction — `prune` snaps mined expansion *words* to an
      edit-distance-1 neighbour in the system word list (`/usr/share/dict/words`
      if present, else skipped). Nothing bundled

### Candidate provenance & the watch list

- [x] provenance `declared` (via `ae watch` or `ae add ACR` with no expansion)
      vs `seen` (auto-detected). An acronym joins the **watch list** (cross-text
      mining) once declared or seen `count` >= `WATCH_THRESHOLD` (3); `prune`
      keeps declared candidates, drops seldom-seen `seen` ones
- [x] `ae candidates` shows provenance (`declared`/`seen`) + watch state
- [x] consolidation = the unified `Store::consolidate` (spell-fix + dedup →
      quality; drop low-conf + clear noise → cleanup), shared by `ae prune` and
      the auto-job. Cadence-gated (`AE_CONSOLIDATE_SECS`, default daily; negative
      disables) via a `meta` table, so dedup/spell run regularly to lift
      confidence; runs in-process and in the warm daemon
- [x] prune grace — a candidate seen within `AE_PRUNE_GRACE_SECS` (default ~30
      days, low volume → patient; `0` = immediate) is spared, so an infrequent
      token isn't yanked before it can recur weeks later. Same grace gates the
      low-confidence drop of *mined* expansions (`recent_potentials`), and
      `dedup_potentials` keeps the newest `last_seen` so merging doesn't reset age
- [ ] tune `WATCH_THRESHOLD` / `AE_GC_PERCENT` / `AE_PRUNE_GRACE_SECS`

### Speculation — next steps

- [ ] dedup across differing word counts / via embedding similarity (prefix only
      handles same-length phrases today)
- [x] age-based grace in GC (`last_seen`) — recent mined rows spared from the
      low-confidence drop; dedup preserves the newest timestamp
- [x] mine *known* acronyms too — alternative meanings become speculative rows;
      a recurrence of a known expansion folds its context in (strengthening
      contextual confidence) instead of duplicating
- [x] trie-based single-pass mining — `MiningTrie` keys acronyms by their letters
      (`PB&J` → `PBJ`); one DFS walk of the text emits every acronym it spells
      (filler-tolerant via consume/skip branching, precision guard intact),
      replacing the per-acronym rescan. O(text) per analysis, not O(acronyms×text)
- [x] cache the base mining trie (watch list ∪ known) in the `Engine`, keyed on a
      cheap `(known, watch-list)` count signature — rebuilt only when it changes
      (which also catches out-of-band `add`/`rm`/`watch` edits) or ages past a
      5-min backstop. Per-request candidates use a tiny separate trie. Persists
      across the warm daemon's requests; the one-shot path builds it once
- [ ] cap/dedup `acronym_contexts` rows — mining a recurrence of a known
      expansion appends a context embedding every time (`add_context`), so the
      table grows unbounded for frequently-seen acronyms and `Engine::contextual`
      scans all of them. Keep a running mean (like `candidate_contexts`) or cap
      to the most recent/representative N, folded into consolidation
- [ ] fzf preview pane showing where each candidate/phrase was seen

## Feature requests / backlog

- [ ] `vec0` virtual-table storage when `sqlite-vec` is available.
- [ ] Bump `MRL_DIMS` to 128 (better disambiguation for the non-MRL model).
- [ ] `ae --add ACR "Expansion"` to seed the dictionary from the CLI.
- [ ] Confidence calibration for learned candidates from real corpora.
- [ ] Homebrew: validate the formula's ONNX-Runtime (load-dynamic) build and
      first-run model fetch on a clean machine (sandbox network constraints).

## Known bugs

- _(none recorded yet)_
