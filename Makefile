# ae — build / install / test helpers.
#
#   make            - same as `make help`
#   make build      - dev build   → ./target/debug/ae
#   make release    - optimized   → ./target/release/ae
#   make install    - cargo install --path . (→ ~/.cargo/bin)
#   make uninstall  - cargo uninstall acronym-engine
#   make test       - cargo test
#   make lint       - cargo fmt --check && cargo clippy (warnings = errors)
#   make fmt        - cargo fmt
#   make clean      - cargo clean
#
# The embedding model is fetched on first use from the HuggingFace Hub into the
# shared cache (~/.cache/huggingface/hub) — not at build time, and reused across
# tools. The default build statically links a downloaded ONNX Runtime so the
# binary runs with zero setup; Homebrew uses ort-load-dynamic instead. All
# release-profile builds strip symbols/debug info (see [profile.release]).
#
# Note: this machine's cargo came via Homebrew's keg-only rustup and may not be
# on PATH. Either add it (see CLAUDE.md) or run, e.g.:
#   make build CARGO=/opt/homebrew/opt/rustup/bin/cargo

CARGO ?= cargo
BIN   := ae

.DEFAULT_GOAL := help
.PHONY: help build release install uninstall test lint fmt clean

help:
	@echo "ae targets:"
	@echo "  make build      dev build → target/debug/$(BIN)"
	@echo "  make release    optimized build → target/release/$(BIN)"
	@echo "  make install    cargo install --path . (→ ~/.cargo/bin)"
	@echo "  make uninstall  cargo uninstall acronym-engine"
	@echo "  make test       cargo test"
	@echo "  make lint       cargo fmt --check && cargo clippy"
	@echo "  make fmt        cargo fmt"
	@echo "  make clean      cargo clean"

build:
	$(CARGO) build

release:
	$(CARGO) build --release

install:
	$(CARGO) install --path .

uninstall:
	$(CARGO) uninstall acronym-engine

test:
	$(CARGO) test

lint:
	$(CARGO) fmt --check
	$(CARGO) clippy --all-targets -- -D warnings

fmt:
	$(CARGO) fmt

clean:
	$(CARGO) clean
