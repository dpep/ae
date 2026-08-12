# Changelog

Notable changes to `ae`. The CLI surface — flags, output shape, exit codes — is
the public API.

Entries are reconstructed from tags and their release notes, so they summarise
what shipped rather than every commit. Only 0.3.3 onward were tagged; earlier
releases are grouped at the end.

## Unreleased

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
