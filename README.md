# ae — Acronym Engine

Acronym extraction and expansion — local, offline.

`ae` reads the text that flows past it and sorts every acronym into three buckets at once:

1. **expansions** — known acronyms, resolved from the dictionary and ranked.
2. **extractions** — acronyms defined inline (`KPI (Key Performance Indicator)`); the new term is pulled out, and the dictionary grows as it reads.
3. **candidates** — acronym-shaped tokens it doesn't recognize and that aren't defined inline — flagged for you to define.

```sh
$ ae "The OKR review needs a TPS (Transaction Processing System) sign-off before the XYZ ships."
OKR      Objectives and Key Results               expansion  v1.00 c0.50
TPS      Transaction Processing System            extraction 0.95
XYZ      (no expansion)                           candidate
```

Unlike a static glossary, ae grows its own dictionary by watching the stream — extracting inline definitions, and *mining* speculative expansions from prose where word initials spell a watched acronym (no parentheses needed).

## Two scores, not one

Every `(acronym, expansion)` pair carries two independent scores — the idea ae is built on:

- **validity** (`v`) — *is this a real expansion of the acronym?* Set by how the pair was learned: `1.0` when a human verified it, `0.9` for an inline definition, `0.0` for a speculative mined guess.
- **confidence** (`c`) — *is this the meaning here?* Cosine fit of the sentence against the contexts where the expansion has appeared; `0.5` with no evidence yet.

Validity asks whether the expansion is real; confidence asks whether it's right *for this sentence*. A pair can be rock-solid valid (`PT → Part Time`) yet low-confidence in a physical-therapy paragraph. Both ride a provenance continuum by source — `user` (verified) > `inline` > `mined` (speculative) — which drives ranking, what shows where, and what gets pruned.

## Built for pipes and agents

ae's real interface is a pipe: send text on stdin, read structured JSON on stdout — `-j` for a pretty object, `-J` for NDJSON:

```sh
$ printf 'ship the MVP this sprint' | ae -j
{
  "sentence": "ship the MVP this sprint",
  "expansions": [
    {
      "acronym": "MVP",
      "text_slice": "MVP",
      "matches": [
        {
          "expansion": "minimum viable product",
          "validity": 1.0,
          "confidence": 0.5
        }
      ]
    }
  ],
  "extractions": [],
  "candidates": []
}
```

Agents and tools hit org-specific acronyms constantly and can't phone a server for them — ae resolves them locally, in-process, in milliseconds.

## Why local-first

No network calls, ever. The dictionary is a bundled SQLite database
(`$XDG_DATA_HOME/ae/acronyms.db`); embeddings run locally via a quantized ONNX
model, with a deterministic hash-embedder fallback so offline builds still work.
Across concurrent callers, ae elects one in-process **Leader** that holds the
warm state behind a Unix socket while the rest proxy to it — no daemon to manage,
and an idle Leader cleans itself up.

## Install

```sh
make install   # → ~/.cargo/bin/ae
make bundle    # standalone: model baked into one self-contained file
```

The embedding model is fetched once at build time into a user cache (`~/.cache/ae`,
reused across rebuilds — never committed). Everything loads it **externally** from
that cache by default (smaller, faster compiles — and how `ae` ships via
Homebrew). `make bundle` (or `--features bundled-model`) is the special case that
**bakes it into the binary** for a single self-contained file you can hand to
another machine. Offline builds still work — they fall back to a deterministic
hash embedder.

## Usage

```sh
ae "text to scan"          # analyze a string
cat file | ae              # analyze stdin (-j / -J for JSON / NDJSON)
ae -b -f notes.md          # batch: line-by-line, line:col-tagged hits
ae -r "…"                  # read-only: expand known acronyms, never learn
```

Full flags and subcommands: `ae --help`. The learned dictionary persists in
SQLite (`$XDG_DATA_HOME/ae/acronyms.db`); the daemon and the in-process fallback
share it, so what's learned in one call is there for the next.

### Managing the dictionary

Subcommands curate the dictionary directly (no flags needed — they're distinct
from analysis input, which arrives as a quoted argument or via stdin):

```sh
ae add MVP "Minimum Viable Product" "Most Valuable Player"   # add (one or more)
ae list                               # list everything
ae list perf                          # filter by substring of acronym or expansion
ae show KPI                           # expansions of one acronym
ae candidates                         # acronyms seen but undefined, by frequency
ae add PB&J                           # declare a token as an acronym (ae mines its expansion later)
ae suggest MVP                        # speculative expansions, --limit N / --min-confidence
ae define MVP                         # promote interactively (fzf), or pass expansions
ae prune                              # GC: spell-fix + dedup (prefix+fuzzy) + drop noise
```

`-q/--quiet` suppresses normal output everywhere (e.g. `ae "…" -q` silently
learns; `ae add … -q` adds without printing).

Each acronym has a **provenance**: `declared` (you said it's an acronym, via
`ae add ACR` with no expansion) or `seen` (ae noticed it). An acronym joins the
**watch list** — where we hunt its expansions in later text — once it's declared
or has been *seen* enough times (default 3); below that it's noise and `ae prune`
drops it. Punctuated acronyms (`PB&J`, `R&D`, `U.S.A`) are
detected and mined too (the `&`/`.` maps to a skipped filler word), and a longer
match wins over its parts (`PB&J` beats `PB`, maximal munch).

`ae prune` also spell-corrects mined expansions against the **system** word list
(`/usr/share/dict/words`, if present — nothing bundled), so "minimum viabel
product" converges to "minimum viable product" before dedup.

You rarely need to run it by hand: the same pass runs **automatically** on a
cadence after a write — at most once per `AE_CONSOLIDATE_SECS` (default daily; a
negative value disables it). The point is the *quality* half — spell-fix and
dedup pool evidence and lift confidence — so it's worth running regularly; the
deletion half is gentle (a candidate seen within `AE_PRUNE_GRACE_SECS`, default
~30 days, is spared, so an infrequent token can recur weeks later before it's
ever considered noise). The warm daemon amortizes it across requests.

`ae define MVP` with no expansions picks interactively from the mined
suggestions — via `fzf` (multi-select) if installed, else a numbered prompt. An
acronym can hold several expansions, so multi-select is first-class.

`ae prune` is the occasional GC: it merges prefix-duplicate expansions
("min viable product" → "minimum viable product"), drops ones below a confidence
floor (default 0.15), and removes seen-once noise candidates. `ae suggest` keeps
a higher bar by default (0.30) since speculation is noisy; both take
`--min-confidence` to override, and `suggest` takes `-l/--limit N`.

`ae suggest` is the payoff of tracking candidates. When `ae` analyzes text it
mines word-sequences whose initials spell a watched candidate acronym — no
parentheses needed, tolerant of skipped filler words (`OKR` = Objectives *and*
Key Results), and across *subsequent* inputs too (a definition mentioned in a
later sentence that never repeats the acronym still accrues). It counts how often
each phrase recurs and how well its context fits where the acronym is used, then
blends both into a confidence:

```sh
$ ae add FOO                                          # declare it — start watching
ae: now watching FOO for expansions
$ ae "the Foundations Of Onboarding workshop was great"   # initials spell FOO
No acronyms found.
$ ae suggest FOO --min-confidence 0                   # ae mined the phrase
FOO   foundation of onboarding   0.50 (1)
```

Confirm one with `ae define MVP` (interactive) or `ae add MVP "Minimum Viable
Product"` (which clears it from the candidate/suggestion lists). It's heuristic —
short acronyms attract noise, which sinks to the bottom on low confidence and is
hidden by the default threshold.

Removal disambiguates when an acronym has several expansions:

```sh
ae rm MVP            # removes it if there's one expansion; else lists them and stops
ae rm MVP valuable  # substring picks one ("Most Valuable Player")
ae rm MVP --all     # removes every expansion
```

`ae candidates` is fed automatically: every analysis records the acronym-shaped
tokens it couldn't resolve and counts how often you've used them, so you can see
what's worth defining. (Defining one clears it from the list.) All of these
honor `-j`/`-J` too.

Default output is human-readable; `-j`/`-J` switch to JSON/NDJSON. stdin (when
piped) wins; otherwise the positional `TEXT` is used; with nothing to do, `ae`
prints help. stdout carries only data — all logs go to stderr, so `ae … | jq` is
always safe. Every command is machine-friendly: `-j`/`-J` work everywhere, and
`--daemon`/`--stop` emit a `{"status": …}` object in those modes.

`--batch` (or `--file`/`cat file | ae -b`) scans input line by line and
aggregates the findings, each tagged with its `line:col` position — grep-style in
human mode, a flat array of hits in `-j`/`-J`. `--read-only` is the safe path for
untrusted or high-volume input — it expands known acronyms without ever writing
to the dictionary.

`--model` lets you point at any compatible model — an absolute/relative path to a
model directory or `.onnx` file, or a bare name resolved against the model search
dirs (`$AE_MODELS_DIR`, the user cache, `<bin>/../share/ae/models`). With no
flag, `ae` uses the bundled (or cached) model, and falls back to the hash
embedder if none loads.

## How it works

```
input ─▶ file lock ─▶ Leader (UDS server, warm trie + dictionary + embedder)
                  └─▶ Follower (forwards text, pipes back JSON)

evaluation = STAGE 1 expansion (trie scan → dictionary → 64-d MRL vector match)
           + STAGE 2 learning  (rule-based extraction of inline definitions)
```

Embeddings come from **all-MiniLM-L6-v2** (int8-quantized ONNX) run locally via
ONNX Runtime — tokenize, mean-pool, then compress with **Matryoshka
Representation Learning**: the 384-d vector is truncated to its first 64
coordinates and L2-normalized, shrinking the vector store ~6×. See
[docs/SPEC.md](docs/SPEC.md) for the full design and
[docs/ROADMAP.md](docs/ROADMAP.md) for status and deliberate deviations (notably
the model choice and the build-time fetch-and-bundle strategy).

## Development

```sh
cargo test          # unit + integration tests
cargo clippy --all-targets
cargo fmt
```

See [CLAUDE.md](CLAUDE.md) for conventions.

## License

MIT — see [LICENSE.txt](LICENSE.txt).
