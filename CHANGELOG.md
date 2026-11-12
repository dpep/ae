# Changelog

Notable changes to `ae`. The CLI surface — flags, output shape, exit codes — is
the public API.

Entries are reconstructed from tags and their release notes, so they summarise
what shipped rather than every commit. Only 0.3.3 onward were tagged; earlier
releases are grouped at the end.

## Unreleased
- Analysis embeds a line only when something reads the embedding — a known
  expansion to score or a candidate to track. Plain prose skips inference,
  which was most of the daemon's CPU when fed streamed command output.
- Mining skips acronyms with fewer than three letters. `C.S` passed the old
  three-*character* check and collected thousands of junk expansions; 2-letter
  candidates were mined from their own sentence. Existing junk rows age out
  through `ae prune`.
- `ae prune` (and the daily auto-job) now drops mined expansions seen only once
  past the grace window. The confidence floor alone never caught them — with no
  context history coherence reads as 1.0 — so one-offs piled up indefinitely.
  The default `AE_PRUNE_GRACE_SECS` drops from ~30 days to 7. Run `ae prune`
  once after upgrading to clear the backlog.
- `ae prune` spell-correction no longer changes a word's first letter, so it
  can't turn a mined expansion into one that no longer spells its acronym
  (`coding` was being "corrected" to `boding`).
- A spell-corrected mined expansion keeps its age. Correcting one used to mark
  it as just seen, sparing it from pruning for another grace window.
- The daily consolidation no longer stalls the daemon for seconds on an
  acronym with thousands of mined expansions, which timed out callers mid-run
  (`Broken pipe` in the daemon log).
  The request that triggers it now gets its reply first, and `ae --status`
  answers even while the daemon is mid-request.
- Each analysis no longer scans every mined expansion and every seen-once
  candidate to check whether the dictionary changed. On a dictionary with tens
  of thousands of mined rows those scans were most of the per-line cost. The
  indexes are added on first open.
- A running daemon now expands acronyms added with `ae add` / `ae define`
  from another shell. Before, its in-memory index was built once at startup,
  so they went unexpanded until it restarted.
- On a dictionary first created before 0.3.3, newly mined expansions were
  stored with no "last seen" time, so pruning never gave them the grace window
  and dropped them at the next pass.

## 0.7.0 — 2026-11-05
- `ae --stop` returns once the daemon has exited, not just acknowledged. Before,
  `ae --stop && ae --daemon` could fail with "daemon exited during startup" (or
  blame a stale daemon) because the old daemon still held the lock while it
  drained.
- `ae --status` reports `served`: how many analyses this daemon has answered.
  Checked against how often the caller thinks it ran, it is the one signal that
  says whether the daemon is being used or every call is quietly falling back
  to a private engine at ten times the memory — which is invisible otherwise.
- A daemon that fails to start now says why. Its stderr goes to a log beside the
  socket (`/tmp/ae.log`), appended across daemons and capped at 1MiB
  (`AE_LOG_MAX_BYTES`, enforced while it runs, not only at startup), instead of
  `/dev/null` — a
  detached process that died silently could not be debugged at any verbosity,
  from anywhere. It logs at info by default, and `RUST_LOG` and `RUST_BACKTRACE`
  reach it, since the daemon inherits the environment that spawned it. Each
  daemon names its version and pid as it starts, so one file covers many lives.
- A caller connecting to check whether the daemon is up no longer logs as a
  failed connection. It happens once per call, so the log the previous entry
  added would have been mostly warnings about nothing.
- "daemon exited during startup" now distinguishes its two causes. Something
  holding the lock without serving — a stale daemon, the one case a retry never
  resolves — is named as such, with the file and the way out; anything else
  points at the log. A caller that lost the election to a daemon that *is*
  coming up now waits for it rather than reporting a failure.

## 0.6.2 — 2026-08-21
- `-d` now streams piped stdin and `--file` through the daemon instead of
  opening a private engine per invocation: ~12MB per call rather than ~120MB,
  because the model is loaded once, by the daemon. Anything calling `ae` on
  every command's output was paying a model load every time.
- Every daemon round trip is bounded: 2 seconds to get the request in, then 5
  seconds for the answer (`AE_CLIENT_TIMEOUT_SECS`). A daemon that stopped
  answering used to hang its caller indefinitely; callers now give up and
  evaluate in-process.
- A daemon whose socket file is deleted out from under it now exits. Nothing can
  wake an accept loop with no socket, so it sat there forever holding the lock —
  unreachable by `ae --stop`, and forcing every later call to evaluate
  in-process. `AE_IDLE_SECS` and `AE_CLIENT_TIMEOUT_SECS` are floored at 1: zero
  meant a daemon that died on its first tick, and a socket timeout std rejects.
- The one-off model download is bounded (`AE_FETCH_TIMEOUT_SECS`, 60s) and the
  daemon's watchdog now covers the model load itself. `hf-hub` builds its HTTP
  client with no timeouts, so a stalled connection — captive portal, dropped
  VPN — hung the load with the lock held and no watchdog yet running: the one
  state no client-side timeout could rescue. A cached model never touches the
  network, so this only bounds the first fetch.
- The idle timeout can no longer be held open indefinitely. It only ran while
  nothing was in flight, so one client that connected and never sent its request
  kept the daemon alive for good — and the Leader would wait on that client
  forever. Both sides are now bounded, and idle is measured by elapsed time.
- The daemon loads its engine before binding the socket, so a reachable socket
  means a daemon that can answer. A caller whose spawned daemon loses the lock
  election also gives up immediately, instead of waiting out the full 3-second
  startup window on a socket that will never appear.

## 0.6.1 — 2026-08-12
- Underscores no longer split tokens, so `MIN_CONFIDENCE` stays one word rather
  than becoming the candidate `MIN`. A wrapping underscore is still punctuation,
  so markdown emphasis (`_API_`) reads as the acronym.
- A daemon serving a different database than `--db`/`$AE_DB` asked for is no
  longer used: the request is evaluated in-process against the database you
  named, with a warning. It previously answered from its own dictionary,
  silently. `ae --status` now reports which database the daemon holds.

## 0.6.0 — 2026-08-11
- Mining credits same-sentence acronym/expansion co-occurrences.
- `rm --all` / `rm --restore` wipe the dictionary, taking an automatic backup
  first.
- Acronyms can be ignored/muted, with a guard against all-caps candidate
  floods; `add` un-mutes an ignored acronym when it's explicitly defined.
- Expansion findings carry `source` and `verified`, matching `ae list`. A
  consumer deciding whether to act on an expansion can now tell a curated entry
  from a mined guess; confidence alone never distinguished them.
- Candidates are no longer mined from identifiers, paths and shell variables.
  `CLAUDE_PLUGIN_ROOT` was being split into `CLAUDE`/`PLUGIN`/`ROOT`, `$HOME`
  into `HOME` and `SKILL.md` into `SKILL` — every fragment acronym-shaped. If
  you have streamed command output into `ae`, expect existing junk candidates
  to remain (nothing is deleted); `ae ignore` mutes them individually.

## 0.5.3 — 2026-06-24
- Pure-Rust regex backend (`fancy-regex`), dropping the Oniguruma C dependency
  and with it the C compile from the Homebrew build.

## 0.5.2 — 2026-06-24
- Output folds validity into a single exposed confidence. Findings carried only
  context-fit confidence, so a human-verified expansion and an inline-learned
  one were indistinguishable at equal fit, and consumers had to know about a
  separate validity axis to tell them apart.

## 0.4.0 — 2026-06-23
- Release 0.4.0.

## 0.3.3 — 2026-06-22
- Drop "real-time" from the crate description.

## 0.1.0 – 0.3.2 — 2026-06
- Initial releases: the core engine (CLI, IPC, storage, trie, MRL compression,
  learning), end-to-end and daemon integration tests, docs, CI, and build
  tooling.
