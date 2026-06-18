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
  -f, --format <FMT>   human | json | ndjson        [default: human]
  -m, --model <SPEC>   embedding model: a path (dir or .onnx) or a name
  -d, --daemon         start a detached background leader
      --stop           stop the running background leader
      --socket <PATH>  UDS path                      [default: /tmp/ae.sock]
  -v, --verbose        engine telemetry to stderr
```

stdin (when piped) wins; otherwise the positional `TEXT` is used. stdout carries
only data — all logs go to stderr, so `ae … | jq` is always safe.

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
