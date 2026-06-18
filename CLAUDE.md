# ae development conventions

`ae` (Acronym Engine) is an ultra-lightweight, local-first CLI + background
service for real-time acronym expansion and definition extraction. Read
[README.md](README.md) for the pitch, [docs/SPEC.md](docs/SPEC.md) for the
design contract, and [docs/ROADMAP.md](docs/ROADMAP.md) for what ships when and
why anything deviates from the spec.

## First principles (do not drift from these)

- **stdout is for data, stderr is for logs.** A consumer piping `ae` must get
  clean JSON/text on stdout. All logging goes through `env_logger` to stderr.
- **Lightweight and local-first.** Idle footprint stays small; no network calls;
  no mandatory model download. Heavy/optional capability (real ONNX inference)
  is feature-gated, never on the default path.
- **Single binary, three roles.** The same binary is CLI, Leader daemon, and
  Follower proxy. A file lock picks the role; nothing else coordinates them.
- **Every command is agent/script-friendly.** *All* output — analysis *and*
  command status (`--daemon`, `--stop`, errors) — honors the format. Default is
  human; `-j/--json` and `-J/--ndjson` select JSON/NDJSON. Resolve the effective
  format once via `Cli::format()`; render analysis through `output::render`
  (or `output::render_lines` for batch) and command results through
  `status`/`fail`. `json` is a pretty object, `ndjson` is one compact object per
  line. When you add a command or a payload field, give it structured output in
  the same change and keep field names stable across `src/types.rs`.
- **Read-only is a first-class mode.** `--read-only` (`-r`) expands without
  extracting or persisting — `Engine::expand_only`, no DB writes. Anything that
  mutates the dictionary must be gated by it.
- **Batch is line-oriented.** `--batch` / `--file` analyze input line by line and
  emit aggregated, `line:col`-tagged hits (`output::render_lines`). A bare
  invocation with no input prints `--help`, not an error.
- **Deterministic where it can be.** Parsing, the trie, MRL compression, and the
  hash fallback embedder are all deterministic and unit-tested. Keep the
  default path working offline — the real model is optional, never required.

## Language and toolchain

Rust, single static binary. `rusqlite` (bundled SQLite, WAL) for storage,
`clap` for the CLI, `regex` for the learning patterns, `fs2` for the file lock,
`ort` (ONNX Runtime, statically linked) + `tokenizers` for embeddings.

The embedding model (`all-MiniLM-L6-v2`, int8-quantized ONNX) is **not in the
repo** — `build.rs` fetches it at build time into `~/.cache/ae` (reused across
builds). The default `bundled-model` feature bakes it into the binary (one
self-contained file); `--no-default-features` loads it externally from the cache
(dev/test — faster compiles). Offline builds fall back to the deterministic
`HashEmbedder`. Never commit a model artifact (`.gitignore` blocks `*.onnx`).

This machine's Rust came via Homebrew's keg-only `rustup`, so `cargo` may not be
on `PATH`. Either add it once —

```sh
echo 'export PATH="/opt/homebrew/opt/rustup/bin:$PATH"' >> ~/.bash_profile
```

— or invoke directly: `/opt/homebrew/opt/rustup/bin/cargo`.

## Repo layout

```text
ae/
  Cargo.toml
  build.rs       ← fetch/cache + (optionally) stage the embedding model
  src/
    main.rs      ← thin entry → cli::run()
    lib.rs       ← module wiring
    cli.rs       ← Cli/Format, input resolution, role dispatch, run()
    types.rs     ← AnalysisPayload and friends (the serialized contract)
    mrl.rs       ← Matryoshka truncate + L2 normalize + cosine
    trie.rs      ← thread-safe prefix tree
    store.rs     ← SQLite schema, dictionary, vector store
    embed.rs     ← Embedder trait, HashEmbedder, default_embedder
    embed/onnx.rs← OnnxEmbedder: model resolution, inference, mean-pool
    learn.rs     ← rule-based acronym/definition extraction
    engine.rs    ← in-process evaluation (expansion + learning)
    ipc.rs       ← lock, Leader server, Follower proxy, daemon, janitor
  docs/          ← SPEC.md (contract), ROADMAP.md (tracker)
  tests/         ← integration tests + the CLI e2e harness
```

Keep it a single crate until there's a concrete reason to split.

## Building, testing, linting

```sh
make build                  # dev build (external model) → target/debug/ae
cargo run -- "KPI (Key Performance Indicator)"
make test                   # unit + integration tests (external model)
make lint                   # fmt --check + clippy (warnings = errors)
cargo fmt                    # format — run before committing
```

Dev/test use `--no-default-features --features ort-download` (external model,
still a statically-linked downloaded ONNX Runtime) for speed — prefer the `make`
targets, which set it. The default build also bundles the model; Homebrew builds
with `--features ort-load-dynamic` (dlopen the keg's ORT at runtime). CI lints/
tests in dev mode and does a bundled `--release` build to prove the single-binary
path compiles.

Before committing: `cargo fmt && cargo clippy --all-targets --no-default-features
--features ort-download -- -D warnings && cargo test --no-default-features
--features ort-download`.

## Testing conventions

- Write tests for new code, focused on quality not quantity — edge cases and
  error handling over restating the happy path.
- **Verify through `cargo test`, not by hand-running the binary.** CLI behavior
  lives in `tests/cli_e2e.rs`, which drives the built binary (`CARGO_BIN_EXE_ae`)
  with an isolated socket/DB in a temp dir — reproducible and CI-checked.
- Use generic, non-identifying test data (`KPI`, `Widget`, `Foo`). This is a
  public repo.
- Spec descriptions stay simple and resilient ("raises an error", not a brittle
  exact-string match).

## Schema changes

`store.rs` owns the schema. A schema change updates the schema block in
[docs/SPEC.md](docs/SPEC.md) (or notes the deviation in ROADMAP) in the same
change.

## Landing changes

Solo project — commit directly to `main` and push. Keep changes small, focused,
and logically connected; change behavior or structure, not both at once. Make
sure CI is green (`cargo fmt --check && cargo clippy --all-targets -- -D warnings
&& cargo test`) before pushing. Update [docs/ROADMAP.md](docs/ROADMAP.md) when a
milestone box moves.
