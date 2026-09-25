//! The engine as one source file.
//!
//! The workspace stays the source of truth; this writes `dist/ecg_engine.rs`
//! from it. Each engine crate becomes a module of the same name, every
//! out-of-line `mod x;` and `include!` is inlined, and paths are rewritten so
//! that `crate::` inside a crate means that crate's module and `ecg_dsp::` from
//! another crate means `crate::ecg_dsp::`. The standard interface,
//! `ecg-ffi`, comes along as `ecg_ffi`, so the one file compiles with plain
//! `rustc` into the C library that a host loads - and replacing that file, or
//! the library built from it, replaces the engine.
//!
//! `--check` writes nothing and fails if the file on disk is not what the
//! workspace would generate, so the two cannot drift apart unnoticed.

use crate::Opts;
use std::path::{Path, PathBuf};

/// Dependency order, which is also the order they appear in the file.
const CRATES: [&str; 7] = [
    "ecg-dsp",
    "ecg-qrs",
    "ecg-quality",
    "ecg-beats",
    "ecg-rhythm",
    "ecg-pipeline",
    "ecg-ffi",
];

fn module_name(krate: &str) -> String {
    krate.replace('-', "_")
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Rewrite `crate::` to `crate::<own>::`, and `<other>::` to
/// `crate::<other>::`, wherever each starts a path.
fn rewrite(line: &str, own: &str) -> String {
    let mut out = String::with_capacity(line.len() + 16);
    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;
    let starts_path = |i: usize| -> bool {
        if i == 0 {
            return true;
        }
        let p = chars[i - 1];
        !(is_ident(p) || p == '$' || p == ':')
    };
    let at = |i: usize, s: &str| -> bool {
        let s: Vec<char> = s.chars().collect();
        chars.len() >= i + s.len() && chars[i..i + s.len()] == s[..]
    };
    let others: Vec<String> = CRATES.iter().map(|c| module_name(c)).collect();
    while i < chars.len() {
        if starts_path(i) && at(i, "crate::") {
            out.push_str("crate::");
            out.push_str(own);
            out.push_str("::");
            i += "crate::".len();
            continue;
        }
        let mut hit = false;
        if starts_path(i) {
            for o in &others {
                let pat = format!("{o}::");
                if at(i, &pat) {
                    out.push_str("crate::");
                    out.push_str(&pat);
                    i += pat.chars().count();
                    hit = true;
                    break;
                }
            }
        }
        if !hit {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// `[pub[(crate)]] mod name;` on a line of its own.
fn out_of_line_mod(line: &str) -> Option<(String, String)> {
    let t = line.trim();
    let (vis, rest) = if let Some(r) = t.strip_prefix("pub(crate) mod ") {
        ("pub(crate) ", r)
    } else if let Some(r) = t.strip_prefix("pub mod ") {
        ("pub ", r)
    } else if let Some(r) = t.strip_prefix("mod ") {
        ("", r)
    } else {
        return None;
    };
    let name = rest.strip_suffix(';')?;
    name.chars()
        .all(is_ident)
        .then(|| (vis.to_string(), name.to_string()))
}

fn expand(file: &Path, children: &Path, own: &str, out: &mut String) -> std::io::Result<()> {
    let text = std::fs::read_to_string(file)?;
    for line in text.lines() {
        if let Some((vis, name)) = out_of_line_mod(line) {
            let a = children.join(format!("{name}.rs"));
            let b = children.join(&name).join("mod.rs");
            let path = if a.exists() { a } else { b };
            let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
            out.push_str(&format!("{indent}{vis}mod {name} {{\n"));
            expand(&path, &children.join(&name), own, out)?;
            out.push_str(&format!("{indent}}}\n"));
            continue;
        }
        if let Some(start) = line.find("include!(\"") {
            let rest = &line[start + "include!(\"".len()..];
            if let Some(end) = rest.find("\")") {
                let path = file.parent().unwrap_or(Path::new(".")).join(&rest[..end]);
                expand(&path, children, own, out)?;
                continue;
            }
        }
        out.push_str(&rewrite(line, own));
        out.push('\n');
    }
    Ok(())
}

/// FNV-1a over the generated body: the same sources give the same identity,
/// and any change to them gives a different one, whatever git says.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn workspace_version(root: &Path) -> String {
    std::fs::read_to_string(root.join("Cargo.toml"))
        .ok()
        .and_then(|t| {
            t.lines()
                .skip_while(|l| l.trim() != "[workspace.package]")
                .find_map(|l| {
                    l.trim()
                        .strip_prefix("version = ")
                        .map(|v| v.trim_matches('"').to_string())
                })
        })
        .unwrap_or_else(|| "0.0.0".into())
}

/// The engine's identity for this workspace: its version and a hash of what
/// would be generated.
pub fn identity(root: &Path) -> std::io::Result<String> {
    let body = generate(root, "")?;
    Ok(format!(
        "{} src {:016x}",
        workspace_version(root),
        fnv1a(&body)
    ))
}

pub fn generate(root: &Path, id: &str) -> std::io::Result<String> {
    let mut out = String::new();
    out.push_str(&format!(
        "//! live-ecg engine, {id}: every engine crate in one file.\n\
         //!\n\
         //! Generated by `ecg-eval amalgamate` from the workspace, which is the source\n\
         //! of truth. Do not edit by hand.\n\
         //!\n\
         //! Rust: build it as its own crate - a `[lib] path` pointing at this file,\n\
         //! named `ecg` - and use `ecg::ecg_ffi::Engine`, or the crates' own types\n\
         //! under `ecg::ecg_pipeline` and the rest. Its paths start at `crate::`, so\n\
         //! it is a crate, not a module to paste into another one.\n\
         //! C and everything else: build it into the standard-interface library,\n\
         //!\n\
         //! ```text\n\
         //! rustc --edition 2021 -O -C panic=unwind --crate-name ecg \\\n\
         //!       --crate-type cdylib --crate-type staticlib ecg_engine.rs\n\
         //! ```\n\
         //!\n\
         //! and use it through `ecg.h`. Swapping this file, or the library built from\n\
         //! it, swaps the engine; a host checks `ecg_abi_version()` and nothing else.\n\
         #![allow(dead_code, unused_imports, clippy::all)]\n\n"
    ));
    for krate in CRATES {
        let own = module_name(krate);
        let src = root.join("crates").join(krate).join("src");
        out.push_str(&format!("pub mod {own} {{\n"));
        let mut body = String::new();
        expand(&src.join("lib.rs"), &src, &own, &mut body)?;
        if krate == "ecg-ffi" {
            let marker = "pub const ENGINE_ID: &str = \"live-ecg (workspace build)\\0\";";
            assert!(body.contains(marker), "ecg-ffi's ENGINE_ID line moved");
            body = body.replace(
                marker,
                &format!("pub const ENGINE_ID: &str = \"live-ecg {id}\\0\";"),
            );
        }
        out.push_str(&body);
        out.push_str("}\n\n");
    }
    Ok(out)
}

pub fn run(opts: &Opts) -> std::io::Result<()> {
    let root = PathBuf::from(opts.get_str("root").unwrap_or("."));
    let path = PathBuf::from(opts.get_str("out").unwrap_or("dist/ecg_engine.rs"));
    // The identity is a hash of the generated sources, so `--check` compares
    // everything, the identity line included.
    let id = match opts.get_str("id") {
        Some(id) => id.to_string(),
        None => identity(&root)?,
    };
    if opts.has("check") {
        let on_disk = std::fs::read_to_string(&path)?;
        let fresh = generate(&root, &id)?;
        if fresh != on_disk {
            eprintln!(
                "{} is stale: run `ecg-eval amalgamate` and commit it",
                path.display()
            );
            std::process::exit(1);
        }
        eprintln!("{} matches the workspace", path.display());
        return Ok(());
    }
    let text = generate(&root, &id)?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&path, &text)?;
    eprintln!(
        "wrote {} ({} lines), {id}",
        path.display(),
        text.lines().count()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_rewritten_only_where_they_start() {
        assert_eq!(
            rewrite("use crate::gbdt::{GbdtModel, Node};", "ecg_beats"),
            "use crate::ecg_beats::gbdt::{GbdtModel, Node};"
        );
        assert_eq!(
            rewrite("use ecg_dsp::{Biquad, Ring};", "ecg_qrs"),
            "use crate::ecg_dsp::{Biquad, Ring};"
        );
        assert_eq!(
            rewrite(
                "let q: ecg_qrs::DetectorState = ecg_qrsx::y;",
                "ecg_pipeline"
            ),
            "let q: crate::ecg_qrs::DetectorState = ecg_qrsx::y;"
        );
        assert_eq!(rewrite("my_ecg_dsp::x", "ecg_qrs"), "my_ecg_dsp::x");
    }

    #[test]
    fn out_of_line_modules_are_recognised() {
        assert_eq!(
            out_of_line_mod("pub mod vf;"),
            Some(("pub ".into(), "vf".into()))
        );
        assert_eq!(
            out_of_line_mod("    mod biquad;"),
            Some(("".into(), "biquad".into()))
        );
        assert_eq!(out_of_line_mod("pub mod trees {"), None);
    }
}
