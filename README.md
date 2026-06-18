# ae — Acronym Engine

Ultra-lightweight, local-first acronym expansion and definition extraction for
the command line and for LLM processes that need real-time jargon resolution.

Feed it text; it does two things at once:

1. **Expansion** — finds known acronyms and returns their expansions, ranked.
2. **Learning** — spots acronyms *defined inline* (`KPI (Key Performance
   Indicator)`) and extracts the new term so the dictionary grows as it reads.

```sh
$ ae "Our KPI (Key Performance Indicator) gates the OKR review."
KPI  Key Performance Indicator        (learned, 0.95)
OKR  Objectives and Key Results       (expansion, 0.80)

$ cat notes.md | ae --format json
{ "sentence": "...", "expansions": [...], "learned_candidates": [...] }
```

## Why

LLM sessions and terminal pipelines hit unfamiliar acronyms constantly. `ae`
resolves them locally — no network, tiny footprint — and gets faster across
concurrent callers by electing one in-process **Leader** that holds the warm
state while everyone else proxies to it over a Unix domain socket. No daemon to
manage: the first caller becomes the leader, the rest follow, and an idle leader
cleans itself up.

## Install

```sh
make install          # cargo install --path . → ~/.cargo/bin/ae
# or
cargo build --release # → target/release/ae
```

## Usage

```text
ae [TEXT] [OPTIONS]

  TEXT                 text to scan; optional when piping via stdin
  -f, --format <FMT>   human | json | ndjson        [default: human]
  -d, --daemon         start a detached background leader
  -s, --stop           stop the running background leader
      --socket <PATH>  UDS path                      [default: /tmp/ae.sock]
  -v, --verbose        engine telemetry to stderr
```

stdin (when piped) wins; otherwise the positional `TEXT` is used. stdout carries
only data — all logs go to stderr, so `ae … | jq` is always safe.

## How it works

```
input ─▶ file lock ─▶ Leader (UDS server, warm trie + dictionary + embedder)
                  └─▶ Follower (forwards text, pipes back JSON)

evaluation = STAGE 1 expansion (trie scan → dictionary → 64-d MRL vector match)
           + STAGE 2 learning  (rule-based extraction of inline definitions)
```

Embeddings are compressed with **Matryoshka Representation Learning**: a 384-d
vector is truncated to its first 64 coordinates and L2-normalized, shrinking the
vector store ~6× while keeping most of the semantic signal. See
[docs/SPEC.md](docs/SPEC.md) for the full design and
[docs/ROADMAP.md](docs/ROADMAP.md) for status and deliberate deviations (notably:
the default embedder is a deterministic hash so nothing has to download a model;
real ONNX inference is feature-gated future work).

## Development

```sh
cargo test          # unit + integration tests
cargo clippy --all-targets
cargo fmt
```

See [CLAUDE.md](CLAUDE.md) for conventions.

## License

MIT — see [LICENSE.txt](LICENSE.txt).
