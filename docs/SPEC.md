---
name: SPEC
description: The original ae (Acronym Engine) system specification, verbatim — the design contract the implementation tracks against.
---

# ae (Acronym Engine) System Specification

> This is the original specification as delivered. It is the design contract.
> Where the implementation deviates (e.g. deferring ONNX inference behind a
> trait, or storing vectors in a plain table instead of `vec0`), the deviation
> is recorded in [ROADMAP.md](ROADMAP.md) with the rationale — not by silently
> editing this document.

## Architectural Overview & Goal

The goal of `ae` is to provide an ultra-lightweight, local-first, zero-dependency
command-line utility and background service that running LLM processes (like
Claude Desktop) or terminal pipes can use to perform real-time acronym expansion
and definition extraction.

## Core Constraints

**Resource Optimization:** Total memory footprint must sit under 30–40 MB of RAM
at idle. Startup latency must be sub-millisecond when running as a client proxy,
and sub-300ms when executing a standalone fallback evaluation.

**Vector Dimensionality Pruning:** The system uses Matryoshka Representation
Learning (MRL) to truncate high-dimensional embeddings down to 64 float elements,
compressing the vector payload database footprint by 6x while retaining over 93%
semantic retrieval accuracy.

**Zero Configuration Overlap:** Uses a hybrid single-binary orchestration
architecture via Unix Domain Sockets (UDS) and explicit OS file locks to cleanly
multiplex multiple client contexts into a unified, shared-memory background
server thread automatically.

## Technical Architecture Blueprint

```
                      ┌──────────────────────────────────────┐
                      │    CLI / IPC INGESTION SUB-SYSTEM    │
                      └──────────────────┬───────────────────┘
                                         │
                    Is stdin an interactive TTY terminal?
                        /                         \
                 [YES] /                           \ [NO]
                      ▼                             ▼
        Extract Explicit Arg Block        Consume Newline-Delimited
          `ae "Some text..."`                Pipe (`cat file | ae`)
                      │                             │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      Acquire `flock` on `/tmp/ae.lock`
                        /                         \
                 [LOCK ACQUIRED]               [LOCK DENIED]
                      /                             \
                     ▼                               ▼
         ┌────────────────────────┐      ┌────────────────────────┐
         │     LEADER NODE        │      │    FOLLOWER PROXY      │
         ├────────────────────────┤      ├────────────────────────┤
         │ • Spawns UDS Socket    │◄─────┤ • Forward Raw Text     │
         │ • Warm Model In Memory │      │ • Wait for Sync JSON   │
         │ • Active RWLock Trie   │      │ • Pipe Output to Stdout│
         └───────────┬────────────┘      └────────────────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ HYBRID EVALUATION STACK   │
       ├───────────────────────────┤
       │ STAGE 1: EXPANSION        │
       │  └── Scan Trie Tree       │
       │  └── Extract Matryoshka   │
       │  └── Match 64d SQLite Vector
       │                           │
       │ STAGE 2: LEARNING         │
       │  └── Rule-Based Regex AST │
       │  └── Isolate New Terms    │
       └───────────────────────────┘
```

## Milestones

### Milestone 1: CLI Configuration, Target Stream Isolation, & Piping

Handle terminal argument parsing, separate standard logging pipelines from
computational data pipelines, and intercept streaming inputs without deadlocking
the process thread.

- CLI struct: `text`, `--daemon`, `--stop`, `--status`, `--format
  {human,json,ndjson}`, `--socket`, `--verbose`.
- `--status` probes the daemon read-only (never starts one) and reports its
  version, pid, uptime, and active embedder; exits non-zero when none is up, so
  `--status -q` is a silent health check.
- Stream splitting: if stdin is not a TTY, read it to a string; else use the
  positional `text` argument; else error.
- `stdout` stays pristine for data; logs go to `stderr` via `env_logger`.

### Milestone 2: IPC Socket Multiplexing & Self-Healing Guardrails

Single-binary multi-process coordinator using safe filesystem locks, preventing
resource collisions across duplicate client windows.

- `flock` on `/tmp/ae.lock` decides Leader vs Follower.
- Leader spawns a UDS listener; Followers forward raw text and pipe back JSON.
- Lazy janitor: when the connection counter hits 0 and stdio disconnects, a
  15-second timer (re-armed by new connections) removes the socket and exits.

### Milestone 3: Embedded Storage Engine & 64-Dimensional MRL Compression

Embed SQLite, optimize the indexing schemas, and implement the normalization
needed to prune 384-dimensional vectors down to 64 dimensions.

- `acronym_dictionary(id, acronym, expansion, created_at)` with a unique index
  on `(acronym, expansion)`.
- A 64-dimensional L2-normalized context-embedding store keyed by acronym.
- `compress_matryoshka_vector`: slice the first 64 coordinates, L2-normalize,
  guard divide-by-zero.

### Milestone 4: Multi-Threaded Trie & Local Model Evaluation

Pair an in-memory prefix tree (Trie) for rapid keyword checks with a local
embedding runtime.

- `TrieNode { children, is_acronym }`, `SharedTrie { root: RwLock<TrieNode> }`.
- Payloads: `MatchCandidate`, `ExpansionResult`, `LearnedCandidate`,
  `AnalysisPayload`.
- Embedding: load a lightweight model (e.g. `nomic-embed-text-v1.5.onnx`),
  tokenize, extract the 384-d embedding, truncate via Milestone 3.

### Milestone 5: Rule-Based Learning Engine & Fallback Routing

Parse sentences to identify new acronyms, assemble the unified payload, and run
standalone if the daemon is missing.

- Pattern Alpha: `(?P<acronym>[A-Z]{2,6})\s\((?P<definition>[A-Za-z\s]{4,60})\)`
  → e.g. `KPI (Key Performance Indicator)`.
- Pattern Beta: `(?P<definition>[A-Za-z\s]{4,60})\s\((?P<acronym>[A-Z]{2,6})\)`
  → e.g. `Key Performance Indicator (KPI)`.
- Fallback: try to connect to the UDS; on failure, hydrate an in-process engine
  (SQLite + Trie + embedder), evaluate locally, render, and exit.
