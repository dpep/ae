//! ae — Acronym Engine.
//!
//! A lightweight, local-first acronym expansion and definition-extraction
//! engine. The same binary plays three roles, picked by a file lock: a CLI, a
//! warm Leader daemon, and a Follower proxy. When no Leader is running, callers
//! self-heal by evaluating in-process. See `docs/SPEC.md` for the design.

pub mod cli;
pub mod embed;
pub mod engine;
pub mod ipc;
pub mod learn;
pub mod mrl;
pub mod output;
pub mod spell;
pub mod store;
pub mod trie;
pub mod types;
