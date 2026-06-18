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
- [x] stream splitting: non-TTY stdin → read to string; else positional arg; else error
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
- [x] model fetched at build time into a reused user cache (`build.rs`), never
      committed; `bundled-model` feature bakes it into one self-contained binary
      (default), or load externally with `--no-default-features`
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
   quantizer at build time needs Python's onnxruntime tooling, which we won't
   add as a build dependency. Downloading the already-quantized artifact gives
   the same size win without it.

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

## Feature requests / backlog

- [ ] `vec0` virtual-table storage when `sqlite-vec` is available.
- [ ] Bump `MRL_DIMS` to 128 (better disambiguation for the non-MRL model).
- [ ] `ae --add ACR "Expansion"` to seed the dictionary from the CLI.
- [ ] Confidence calibration for learned candidates from real corpora.
- [ ] Homebrew: validate the formula's ONNX-Runtime + bundled-model build on a
      clean machine (sandbox network constraints).

## Known bugs

- _(none recorded yet)_
