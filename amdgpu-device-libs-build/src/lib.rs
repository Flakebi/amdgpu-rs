// Update: cargo readme > README.md

#![allow(clippy::needless_doctest_main)]
//! Build script support for `amdgpu-device-libs`.
//!
//! Adds linker flags to link device-libs.
//! Add `amdgpu-device-libs-build` as a `build-dependency` and call it in the build script.
//! ```rust,no_run
//! // build.rs
//! fn main() {
//!     amdgpu_device_libs_build::build();
//! }
//! ```
//!
//! This link to the [ROCm device-libs](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs) and a pre-compiled helper library.
//! The libraries are linked from a ROCm installation.
//! To make sure the libraries are found, set the environment variable `ROCM_PATH` or `ROCM_DEVICE_LIB_PATH` (higher priority if it is set).
//! It looks for `amdgcn/bitcode/*.bc` files in this path.
//! See the documentation of [`amdgpu-device-libs`](https://docs.rs/amdgpu-device-libs) for more information.

use std::collections::HashSet;

use rustflags::Flag;

#[derive(Default)]
pub struct Build {
    pub link_args: Vec<String>,
    pub used_env_vars: Vec<String>,
    pub used_files: Vec<String>,
}

pub fn get_link_args(mut is_wave64_enabled: bool, target_cpu: &str) -> Build {
    let mut build = Build::default();

    let cur_dir = env!("CARGO_MANIFEST_DIR");

    let rocm_path = std::env::var("ROCM_PATH").unwrap();
    let rocm_device_lib_path = std::env::var("ROCM_DEVICE_LIB_PATH").unwrap_or(rocm_path);
    let device_libs = format!("{}/amdgcn/bitcode", rocm_device_lib_path);
    build.used_env_vars.push("ROCM_PATH".into());
    build.used_env_vars.push("ROCM_DEVICE_LIB_PATH".into());

    let gfxip = target_cpu
        .strip_prefix("gfx")
        .unwrap_or_else(|| panic!("target-cpu '{target_cpu}' did not start with gfx"));

    build.link_args.push(format!("{device_libs}/ockl.bc"));
    build
        .link_args
        .push(format!("{device_libs}/oclc_isa_version_{gfxip}.bc"));
    build
        .link_args
        .push(format!("{device_libs}/oclc_abi_version_600.bc"));

    // wave64 is the default on gfx9 and before
    is_wave64_enabled |= gfxip.starts_with('9') && gfxip.len() == 3;
    is_wave64_enabled |= gfxip.starts_with("9-") && gfxip.ends_with("-generic");
    let wavesize = if is_wave64_enabled { 64 } else { 32 };
    build.link_args.push(format!(
        "{device_libs}/oclc_wavefrontsize64_{}.bc",
        if is_wave64_enabled { "on" } else { "off" }
    ));

    build
        .used_files
        .push(format!("{cur_dir}/util{wavesize}.bc"));
    build.link_args.push(format!("{cur_dir}/util{wavesize}.bc"));

    // Workarounds to make linker-plugin-lto work
    build.link_args.push("--undefined-version".into());
    build.link_args.push("--no-gc-sections".into());
    build
}

/// Link libraries for `amdgpu-device-libs`.
///
/// Call in a cargo buildscript:
/// ```rust,no_run
/// // build.rs
/// fn main() {
///     amdgpu_device_libs_build::build();
/// }
/// ```
pub fn build() {
    // Find out target cpu and enabled features
    let mut target_features = std::env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect::<HashSet<_>>();

    let mut target_cpu = None;
    for flag in rustflags::from_env() {
        if let Flag::Codegen { opt, value } = flag {
            if opt == "target-cpu" {
                target_cpu = value;
            } else if opt == "target-feature"
                && let Some(feat) = value
            {
                if let Some(feat) = feat.strip_prefix('-') {
                    target_features.remove(feat);
                } else {
                    let feat = feat.trim_start_matches('+');
                    target_features.insert(feat.into());
                }
            }
        }
    }
    let target_cpu = target_cpu.expect("Did not find target-cpu in RUSTFLAGS");
    let is_wave64_enabled = target_features.contains("wavefrontsize64");
    let mut build = get_link_args(is_wave64_enabled, &target_cpu);
    build.used_env_vars.push("CARGO_CFG_TARGET_FEATURE".into());

    for v in &build.used_env_vars {
        println!("cargo::rerun-if-env-changed={v}");
    }
    for f in &build.used_files {
        println!("cargo::rerun-if-changed={f}");
    }
    for a in &build.link_args {
        println!("cargo::rustc-link-arg={a}");
    }
}
