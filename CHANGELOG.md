# Changelog

Notable changes to `ae`. The CLI surface — flags, output shape, exit codes — is
the public API.

Entries are reconstructed from tags and their release notes, so they summarise
what shipped rather than every commit. Only 0.3.3 onward were tagged; earlier
releases are grouped at the end.

## Unreleased
- Candidates skip tokens that aren't acronyms: an ordinary word in caps
  (`WHERE`, `NOT`, `POST`, `WARN` — checked against the system wordlist, when
  one is installed), a numbered label (`V1`, `L4`, `Q1`, `H1`, `P0`, `FY27`),
  and `INFO`/`DEBUG`/`TODO`/`FIXME`. A real acronym that is also a word (`REST`,
  `CAT`) is no longer surfaced as a candidate; define it with `ae add`,
  or `ae add` it with no expansion to keep mining for one. Junk candidates
  already recorded are cleared by `ae prune` (below).
- `ae rm ACR <expansion>` prefers an exact match: `ae rm PP "Partner Platform"`
  no longer refuses because "Agent and Partner Platform" also contains it, and
  `ae rm BK buildkite` picks that exact casing over "Buildkite". Failing an
  exact match, a unique match ignoring case is used, then the old substring rule.
- `ae rm ACR --unverified` drops every expansion of ACR you didn't add yourself
  (inline-learned and speculative mined rows), keeping `user` ones. With no
  acronym it does that across the whole dictionary, backing it up first. If inline learning has filled your dictionary with junk, run
  `ae rm --unverified` to keep only what you added (and the built-in defaults);
  `ae rm --restore` undoes it.
- Expansions that differ only in case are now one row: learning or adding
  `buildkite` next to an existing `Buildkite` no longer creates a sibling. The
  existing row keeps its casing unless a stronger source arrives — `ae add`
  over an inline-learned row upgrades it to yours, casing included. Siblings
  already in your dictionary stay; remove the one you don't want with
  `ae rm ACR <exact expansion>`.
- `rm`'s JSON status now always carries `backup` (a path, or `null`), and
  `acronym` is `null` for a removal spanning every acronym.
- An expansion's `confidence` is now the probability it's the meaning here:
  its share of the contextual fit among the acronym's expansions, times its
  provenance validity, rounded to two decimals. An acronym with one expansion
  scores its validity (1.0 verified, 0.9 inline); siblings split the
  probability between them. It used to be the raw similarity between the
  prompt and the expansion's own words, which sat near zero for most prompts.
  A consumer gating on `confidence` should re-tune its threshold. Junk siblings
  now dilute a verified expansion's share. Run `ae rm --unverified` (or
  `ae rm ACR --unverified`) to clear them and let verified entries score high.
- An inline definition is added to the dictionary only when its initials spell
  the acronym (`KPI (Key Performance Indicator)`). Others, which were mostly
  sentence fragments and names, still show up as extractions but go to
  `ae suggest` as speculation, and their acronym stays a candidate. Unaligned
  entries learned before this change remain in the dictionary as `inline`.
  `ae rm --unverified` clears them.
- Without the embedding model (offline, not yet cached), context now counts:
  the built-in fallback used to drop most words before comparing, so siblings
  split confidence evenly whatever the prompt said. Nothing to do. Contexts
  stored offline before this change still match, just more weakly, and fill
  in as `ae` sees the acronym again.
- `ae prune` (and the periodic auto-prune) now drops recorded candidates that
  aren't acronyms by the new candidate rule, however often they were seen,
  along with the expansions mined for them. Run `ae prune` once to clear the `WHERE`,
  `NOT`, `TODO` and `V1` already on your candidate list. Acronyms you declared
  with a bare `ae add` are kept. Its JSON gains `non_acronyms_removed` and
  `non_acronym_expansions_removed`.
- `ae stats` reports how much of the dictionary is signal: confirmed
  expansions and acronyms, how many are verified (and how many of those are
  the built-in defaults), how many inline-learned ones spell their acronym,
  and the mined and candidate counts. Shares come from those counts, rounded
  to two decimals.
- `rm` backups now live beside the database they protect, in
  `backups/<db name>-<timestamp>.db` (for the default dictionary,
  `~/.local/share/ae/backups/`), readable only by you, keeping the newest 10.
  They used to share `/tmp/ae` across every database, so a bare
  `ae rm --restore` could restore another dictionary's backup over yours, and
  a reboot deleted them. A bare `--restore` now only considers this
  database's backups. Older backups stay in `/tmp/ae`; restore one by passing
  its path (`ae rm --restore /tmp/ae/backup_….db`). A bulk remove that would
  remove nothing no longer takes a backup.
- `ae prune` now brings dictionary entries an older `ae` learned up to the
  current rules, keeping the good ones: it drops inline definitions whose
  initials don't spell their acronym, and merges confirmed expansions that
  differ only in case (yours over an inline one, else the older casing). Run
  `ae prune` once instead of `ae rm --unverified` to clear inline junk without
  losing the inline definitions that are right. It backs up first when it
  changes any entries. Its JSON gains `unaligned_removed`,
  `case_duplicates_merged` and `backup`. The automatic pass doesn't do this.
- The candidate word rule also catches inflections the system wordlist lacks
  (`FAILED`, `VALUES`, `MOVED`, `FIELDS`), keeping acronyms like `AWS`, `CORS`
  and `SAAS` whose stripped form would be a short word. `AI`, `IDE`, `ISO`,
  `ROI`, `ETA` and `SLA` are exempt and stay candidates: their word senses are
  too obscure to be what caps text means. `ae prune` clears the inflected words
  already on your candidate list.
- Piping `ae` into a reader that stops early (`ae candidates | head`) exits
  quietly with success instead of printing "render failed: Broken pipe" (or a
  JSON error) and failing.
- A known acronym expands only when written in acronym casing: `us`, `it`
  and `It` no longer expand as `US` and `IT`, and a mention inside an
  identifier or path (`src/ID/x.rs`, `config.ID.url`, `foo_ID`) is skipped.
  `ae "look at the okr"` no longer expands; write `OKR`. Nothing to do.
- Inline definitions in camel case now count as spelling their acronym:
  `JSON (JavaScript Object Notation)` and `GitHub (GH)` are learned, and
  `ae prune` keeps such rows already in your dictionary. One that
  doesn't spell its acronym is reported as an extraction and nothing more; it
  is not kept for `ae suggest` either. Nothing to do.
- Expansions are mined only from text that mentions the acronym. Mining a
  watched acronym from any text whose initials happened to spell it
  (`UTF: use the foo`) made nearly all of `ae suggest`'s list. A declared
  acronym is now mined where it's written, not everywhere. Mined rows
  already in your dictionary are mostly such coincidences: run
  `ae rm --unverified` to drop them (it also drops inline-learned rows; see
  above), or let `ae prune` age out the ones seen once.
- Confidence between sibling expansions is less extreme: a 0.15 cosine lead
  is now e:1 odds, not 0.1, which on labeled ambiguous prompts makes the top
  pick's confidence match how often it is right. A consumer gating on
  `confidence` for an acronym with several meanings may see slightly lower
  scores. Scoring a sibling with no stored context no longer re-runs the model
  for its phrase on every hit.
- Opening a dictionary drops the `candidate_contexts` table, which scored the
  cross-text mining removed above. Nothing to do; an older `ae` recreates it.
- `ae add ACR` with no expansion now says what watching means: expansions are
  mined from text that mentions ACR. The JSON status is unchanged.
- Slash-joined acronyms (`CI/CD`, `TCP/IP`, `GPU/CPU`) now expand and become
  candidates part by part; they used to be skipped as paths. Real paths
  (`src/main.rs`, `/usr/bin`) are still skipped. Nothing to do.

## 0.7.1 — 2026-11-12
- When the daemon fails to analyze a line it now hangs up instead of replying
  with an empty result, so the caller falls back in-process and reports the
  real error rather than "no acronyms".
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
- `ae rm --restore` stops a daemon serving that dictionary before swapping the
  file in. It used to replace the file and delete the WAL out from under the
  daemon's open connection.

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
