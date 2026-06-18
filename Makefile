# ae — build / install / test helpers.
#
#   make            - same as `make help`
#   make build      - dev build (external model)   → ./target/debug/ae
#   make release    - optimized, model bundled in  → ./target/release/ae
#   make install    - cargo install --path . (bundled, → ~/.cargo/bin)
#   make uninstall  - cargo uninstall ae
#   make test       - cargo test (external model — fast, small test binaries)
#   make model      - fetch + cache the embedding model (no build)
#   make lint       - cargo fmt --check && cargo clippy (warnings = errors)
#   make fmt        - cargo fmt
#   make clean      - cargo clean
#
# The embedding model is fetched at build time into a user cache (~/.cache/ae)
# and reused across rebuilds — never committed. Release/install builds bundle it
# into a single self-contained binary; dev/test builds load it externally from
# the cache (smaller binaries, faster compiles) via --no-default-features.
#
# Note: this machine's cargo came via Homebrew's keg-only rustup and may not be
# on PATH. Either add it (see CLAUDE.md) or run, e.g.:
#   make build CARGO=/opt/homebrew/opt/rustup/bin/cargo

CARGO ?= cargo
BIN   := ae
# Dev/test: external model (faster compiles) but still statically link a
# downloaded ONNX Runtime so it runs with zero setup.
DEV   := --no-default-features --features ort-download

.DEFAULT_GOAL := help
.PHONY: help build release install uninstall test model lint fmt clean

help:
	@echo "ae targets:"
	@echo "  make build      dev build (external model) → target/debug/$(BIN)"
	@echo "  make release    optimized, model bundled in → target/release/$(BIN)"
	@echo "  make install    cargo install --path . (bundled, → ~/.cargo/bin)"
	@echo "  make uninstall  cargo uninstall $(BIN)"
	@echo "  make test       cargo test (external model)"
	@echo "  make model      fetch + cache the embedding model"
	@echo "  make lint       cargo fmt --check && cargo clippy"
	@echo "  make fmt        cargo fmt"
	@echo "  make clean      cargo clean"

build:
	$(CARGO) build $(DEV)

release:
	$(CARGO) build --release

install:
	$(CARGO) install --path .

uninstall:
	$(CARGO) uninstall $(BIN)

test:
	$(CARGO) test $(DEV)

# build.rs fetches the model into the cache during any build; this target warms
# it without producing artifacts.
model:
	$(CARGO) build $(DEV) --quiet
	@echo "model cached under $${XDG_CACHE_HOME:-$$HOME/.cache}/ae/models"

lint:
	$(CARGO) fmt --check
	$(CARGO) clippy --all-targets $(DEV) -- -D warnings

fmt:
	$(CARGO) fmt

clean:
	$(CARGO) clean
