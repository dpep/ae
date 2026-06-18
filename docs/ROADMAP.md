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
- [x] `Embedder` trait; deterministic `HashEmbedder` default (see deviation)
- [~] real ONNX (`nomic-embed-text-v1.5`) embedder — feature-gated, future work
- [x] tests: trie insert/scan, embedding determinism + dimensionality

## Milestone 5 — rule-based learning + fallback routing

- [x] Pattern Alpha / Beta extraction (regex), confidence scoring, dedup
- [x] unified `AnalysisPayload` assembly (expansions + learned candidates)
- [x] in-process evaluation engine (Trie + dictionary + embedder)
- [x] fallback routing: UDS client first, in-process engine on connect failure
- [x] learned candidates persisted back to the dictionary
- [x] tests: both patterns, no false positives, fallback path, persistence

## Deviations from the spec (with rationale)

1. **Embedding inference is behind an `Embedder` trait, default `HashEmbedder`.**
   The spec's `ort` (ONNX Runtime) + `tokenizers` path needs a ~hundreds-of-MB
   model file that can't ship in a clean checkout or CI, and the heavy native
   deps fight the "ultra-lightweight, zero-dependency" goal. The default
   embedder is a deterministic feature-hash that produces a real 384-d vector,
   so the *entire* MRL pipeline (truncate → normalize → store → cosine) is
   genuine and tested. A real ONNX embedder can be added behind a `onnx`
   feature flag without touching callers. Tracked as the one `[~]` above.

2. **Vectors live in a plain `acronym_contexts` table, not a `vec0` virtual
   table.** `sqlite-vec`'s `vec0` needs a loadable extension that `rusqlite`'s
   bundled SQLite doesn't carry. At this corpus size cosine in Rust over the
   candidate set is simpler and fast. Swapping in `vec0` later is a storage-layer
   change behind the same API.

3. **`--stop` and the janitor.** `--stop` connects and sends a shutdown frame;
   the janitor is an idle-timeout watchdog thread. The spec's "stdio
   disconnect" trigger is approximated by an idle-connection timeout, which is
   the portable, testable equivalent.

## Feature requests / backlog

- [ ] Real ONNX embedder behind `--features onnx` (model path via env).
- [ ] `vec0` virtual-table storage when `sqlite-vec` is available.
- [ ] `ae --add ACR "Expansion"` to seed the dictionary from the CLI.
- [ ] `ndjson` streaming for multi-line piped input (one payload per line).
- [ ] Confidence calibration for learned candidates from real corpora.

## Known bugs

- _(none recorded yet)_
