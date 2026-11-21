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

## Matching

A token is a mention of a known acronym only when it is written in acronym
casing: no lowercase letters (`KPI`, `S3`, `PB&J`). Acronyms are stored
uppercase, so casing is the only thing that separates `US` from `us` or `IT`
from `it`. Tokens inside an identifier or path (`src/ID/x.rs`, `config.ID.url`,
`$ID`, `foo_ID`) are skipped, by the same rule that keeps them off the
candidate list.

## Expansion confidence

An expansion finding's `confidence` is the probability that this expansion is
what the acronym means here, discounted by how far its provenance can be
trusted:

```text
confidence = validity × share          rounded to 2 decimals
share_i    = softmax_i(fit / 0.1)      over the acronym's confirmed expansions
fit_i      = max cosine(query, context) over expansion i's recorded contexts,
             or cosine(query, embed(expansion i)) when it has none
```

- **validity**: provenance prior, `user` 1.0, `inline` 0.9.
- **share**: P(expansion | acronym, context). Shares across an acronym's
  expansions sum to 1. A sole expansion has nothing to disambiguate, so its
  share is 1.0 and its confidence equals its validity (verified 1.0, inline
  0.9). Raw cosines aren't probabilities; the temperature (0.1) reads a 0.1
  cosine lead as e:1 odds.
- An expansion with no recorded context is scored against its own phrase, the
  same vector its first context would hold, so it is neither favoured nor
  penalized against siblings that have one.
- Matches are ranked by confidence, with validity breaking ties.
- Only aligned inline definitions (initials spell the acronym) are confirmed as
  `inline`. An unaligned one is still reported as an extraction, and its
  acronym stays a candidate, but it is recorded only as a speculative `mined`
  expansion for `ae suggest`.

## Milestones

### Milestone 1: CLI Configuration, Target Stream Isolation, & Piping

Handle terminal argument parsing, separate standard logging pipelines from
computational data pipelines, and intercept streaming inputs without deadlocking
the process thread.

- CLI struct: `text`, `--daemon`, `--stop`, `--status`, `--format
  {human,json,ndjson}`, `--socket`, `--verbose`.
- `--status` probes the daemon read-only (never starts one) and reports its
  version, pid, uptime, active embedder, and how many analyses it has served;
  exits non-zero when none is up, so `--status -q` is a silent health check.
  The served count against the caller's own is what says whether the daemon is
  being used or bypassed — the in-process fallback is otherwise invisible.
- Input source picks the mode: a positional `text` argument is analyzed as one
  blob; piped stdin and `--file` are streamed line by line (flushed per line for
  human/NDJSON; pretty JSON buffers an array). `-d` sends either through the
  daemon, so a repeated caller loads the model once, not once per call. Either
  way the structured output is the same flat list of findings (`{kind, acronym,
  …}`) — single and stream, `-j` and `-J`, all share one shape. A bare TTY invocation prints help.
- `ae stats` reports dictionary health from counts alone: `confirmed`,
  `acronyms`, `verified` (`user` rows), `builtin` (verified rows still matching
  the seeded defaults; the schema has no seeded marker), `verified_share`,
  `inline`, `aligned` (inline rows whose initials spell the acronym, by
  `learn::is_aligned`), `aligned_share`, `mined`, `candidates`, `watched`.
  Shares are rounded to two decimals where they're computed, and `null` with a
  zero denominator. The hook's inject rate lives with the plugin, not here.
- `stdout` stays pristine for data; logs go to `stderr` via `env_logger`.

### Milestone 2: IPC Socket Multiplexing & Self-Healing Guardrails

Single-binary multi-process coordinator using safe filesystem locks, preventing
resource collisions across duplicate client windows.

- `flock` on `/tmp/ae.lock` decides Leader vs Follower.
- Leader spawns a UDS listener; Followers forward raw text and pipe back JSON.
  The engine is loaded before the socket is bound: a reachable socket means a
  Leader that can answer.
- Neither side ever waits forever. A Follower allows 2s to get its request in
  and `AE_CLIENT_TIMEOUT_SECS` (default 5s) for the answer, then self-heals
  in-process; the Leader bounds its side too, so a stalled client can't pin a
  serving thread.
- The Leader's stderr is a log beside the socket (`<socket>.log`), appended
  across daemons and capped at `AE_LOG_MAX_BYTES` (default 1MiB, newest half
  kept). It detaches, so nothing else can account for why it died — and a crash
  loop is what the log is most needed for, so it must survive a restart. The cap
  is enforced by the janitor as well as at startup: a debug-level run logs twice
  a second, and would otherwise never re-check.
- Lazy janitor: `AE_IDLE_SECS` (default 300s) without a finished request removes
  the socket and exits. In-flight requests don't gate it — that would let one
  stuck client keep the daemon alive forever — they get a bounded drain.

### Milestone 3: Embedded Storage Engine & 64-Dimensional MRL Compression

Embed SQLite, optimize the indexing schemas, and implement the normalization
needed to prune 384-dimensional vectors down to 64 dimensions.

- `acronym_dictionary(id, acronym, expansion, created_at)` with a unique index
  on `(acronym, expansion)` (case-sensitive, but adding an expansion that
  differs only in case updates the existing row instead of inserting a
  sibling), and one on `(source, acronym)` so confirmed-only
  lookups don't scan the (far larger) set of mined rows. Candidate acronyms
  carry a partial index over just the watch list, for the same reason.
- `ignored_acronyms(acronym, created_at)`: a mute list. Ignored acronyms stay in
  the dictionary but go inert — excluded from the expansion trie, the mining and
  watch lists, suggestions, and candidate surfacing — until `unignore`. Distinct
  from `rm`, which deletes the rows.
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
