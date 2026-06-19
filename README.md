# ae — Acronym Engine

Ultra-lightweight, local-first acronym expansion and definition extraction for
the command line and for LLM processes that need real-time jargon resolution.

Feed it text; it sorts the acronyms into three buckets at once:

1. **expansions** — known acronyms, resolved from the dictionary and ranked.
2. **extractions** — acronyms *defined inline* (`KPI (Key Performance
   Indicator)`); the new term is extracted and the dictionary grows as it reads.
3. **candidates** — acronym-shaped tokens it doesn't recognize and that aren't
   defined inline (e.g. `MVP` in "ship the MVP") — candidates for you to define.

```sh
$ ae "Our KPI (Key Performance Indicator) gates the OKR review, then the MVP."
KPI  Key Performance Indicator        extraction 0.95
OKR  Objectives and Key Results       expansion  v1.00 c0.80
MVP  (no expansion)                   candidate

$ cat notes.md | ae -j
{ "sentence": "...", "expansions": [...], "extractions": [...], "candidates": [...] }
```

Expansions carry two scores: **validity** (`v` — is this a real expansion of the
acronym; 1.0 when a human verified it, 0.9 for an inline definition) and
**confidence** (`c` — does it fit *this* sentence, from context vectors). Under
the hood every `(acronym, expansion)` lives on one continuum by `source`:
`user` (verified) > `inline` > `mined` (speculative) — `ae list` shows the
`source`/`verified` of each, `ae suggest` shows the speculative tail.

## Why

LLM sessions and terminal pipelines hit unfamiliar acronyms constantly. `ae`
resolves them locally — no network, tiny footprint — and gets faster across
concurrent callers by electing one in-process **Leader** that holds the warm
state while everyone else proxies to it over a Unix domain socket. No daemon to
manage: the first caller becomes the leader, the rest follow, and an idle leader
cleans itself up.

## Install

```sh
make install          # cargo install --path . → ~/.cargo/bin/ae  (one self-contained binary)
# or
cargo build --release # → target/release/ae
```

The embedding model is fetched once at build time into a user cache (`~/.cache/ae`,
reused across rebuilds — never committed). Release/install builds **bake it into
the binary** so `ae` ships as a single self-contained file; dev/test builds load
it externally for faster compiles (`make build` / `make test`, i.e.
`--no-default-features`). Offline builds still work — they fall back to a
deterministic hash embedder.

## Usage

```text
ae [TEXT] [OPTIONS]

  TEXT                 text to scan; optional when piping via stdin
  -j, --json           JSON output (pretty object)
  -J, --ndjson         NDJSON output (one object per line)
  -b, --batch          analyze input line by line, aggregated with line:col
  -f, --file <PATH>    read input from a file (implies --batch)
  -r, --read-only      expand only — never extract or persist new acronyms
  -m, --model <SPEC>   embedding model: a path (dir or .onnx) or a name
  -d, --daemon         start a detached background leader
      --stop           stop the running background leader
      --db <PATH>      acronym dictionary    [env: AE_DB] [default: data dir]
      --socket <PATH>  UDS path                      [default: /tmp/ae.sock]
  -v, --verbose        engine telemetry to stderr
```

The learned dictionary persists in a SQLite database — by default
`$XDG_DATA_HOME/ae/acronyms.db` (else `~/.local/share/ae/acronyms.db`), or
wherever `--db`/`$AE_DB` points. The daemon and the in-process fallback share it,
so acronyms learned in one invocation are available to the next.

### Managing the dictionary

Subcommands curate the dictionary directly (no flags needed — they're distinct
from analysis input, which arrives as a quoted argument or via stdin):

```sh
ae add MVP "Minimum Viable Product" "Most Valuable Player"   # add (one or more)
ae list                               # list everything
ae list perf                          # filter by substring of acronym or expansion
ae show KPI                           # expansions of one acronym
ae candidates                         # acronyms seen but undefined, by frequency
ae watch PB&J                         # declare a token is an acronym (same as `add PB&J`)
ae suggest MVP                        # speculative expansions, --limit N / --min-confidence
ae define MVP                         # promote interactively (fzf), or pass expansions
ae prune                              # GC: spell-fix + dedup (prefix+fuzzy) + drop noise
```

`-q/--quiet` suppresses normal output everywhere (e.g. `ae "…" -q` silently
learns; `ae add … -q` adds without printing).

Each acronym has a **provenance**: `declared` (you said it's an acronym, via
`ae watch` or `ae add ACR` with no expansion) or `seen` (ae noticed it). An
acronym joins the **watch list** — where we hunt its expansions in later text —
once it's declared or has been *seen* enough times (default 3); below that it's
noise and `ae prune` drops it. Punctuated acronyms (`PB&J`, `R&D`, `U.S.A`) are
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

```
$ ae suggest MVP --min-confidence 0
MVP   minimum viable product    0.50 (2)
MVP   min viable product        0.25 (1)
MVP   most valuable player      0.25 (1)
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
