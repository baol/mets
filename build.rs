//! Records the git commit the crate was built from (METS_GIT_HASH), so the
//! examples' logs say which code produced them.
use std::process::Command;

fn main() {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let hash = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty =
        git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    println!(
        "cargo:rustc-env=METS_GIT_HASH={hash}{}",
        if dirty { "-dirty" } else { "" }
    );
    for path in [".git/HEAD", ".git/index", "src", "examples"] {
        println!("cargo:rerun-if-changed={path}");
    }
    if let Some(head) = git(&["symbolic-ref", "-q", "HEAD"]) {
        println!("cargo:rerun-if-changed=.git/{head}");
    }
}
