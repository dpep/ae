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
- **Every command is agent/script-friendly.** Output honors
  `--format {human,json,ndjson}`. Keep field names stable across the payload
  types in `src/types.rs`.
- **Deterministic where it can be.** Parsing, the trie, MRL compression, and the
  default embedder are all deterministic and unit-tested. Don't make the default
  path depend on an external model or the network.

## Language and toolchain

Rust, single static binary. `rusqlite` (bundled SQLite, WAL) for storage,
`clap` for the CLI, `regex` for the learning patterns, `fs2` for the file lock.

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
  src/
    main.rs      ← thin entry → cli::run()
    lib.rs       ← module wiring
    cli.rs       ← Cli/Format, input resolution, role dispatch, run()
    types.rs     ← AnalysisPayload and friends (the serialized contract)
    mrl.rs       ← Matryoshka truncate + L2 normalize + cosine
    trie.rs      ← thread-safe prefix tree
    store.rs     ← SQLite schema, dictionary, vector store
    embed.rs     ← Embedder trait + deterministic HashEmbedder
    learn.rs     ← rule-based acronym/definition extraction
    engine.rs    ← in-process evaluation (expansion + learning)
    ipc.rs       ← lock, Leader server, Follower proxy, daemon, janitor
  docs/          ← SPEC.md (contract), ROADMAP.md (tracker)
  tests/         ← integration tests + the CLI e2e harness
```

Keep it a single crate until there's a concrete reason to split.

## Building, testing, linting

```sh
cargo build                 # dev build → target/debug/ae
cargo run -- "KPI (Key Performance Indicator)"
cargo test                  # unit + integration tests
cargo clippy --all-targets  # lint — keep it clean
cargo fmt                    # format — run before committing
```

Before committing: `cargo fmt && cargo clippy --all-targets && cargo test`.

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
