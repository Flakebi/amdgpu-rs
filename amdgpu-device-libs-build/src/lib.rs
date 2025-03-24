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
    let cur_dir = env!("CARGO_MANIFEST_DIR");
    println!("cargo::rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");

    let rocm_path = std::env::var("ROCM_PATH").unwrap();
    let rocm_device_lib_path = std::env::var("ROCM_DEVICE_LIB_PATH").unwrap_or(rocm_path);
    let device_libs = format!("{}/amdgcn/bitcode", rocm_device_lib_path);
    println!("cargo::rerun-if-env-changed=ROCM_PATH");
    println!("cargo::rerun-if-env-changed=ROCM_DEVICE_LIB_PATH");
    println!("cargo::rustc-link-arg={}/ockl.bc", device_libs);

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
            } else if opt == "target-feature" {
                if let Some(feat) = value {
                    if let Some(feat) = feat.strip_prefix('-') {
                        target_features.remove(feat);
                    } else {
                        let feat = feat.trim_start_matches('+');
                        target_features.insert(feat.into());
                    }
                }
            }
        }
    }
    let target_cpu = target_cpu.expect("Did not find target-cpu in RUSTFLAGS");
    let gfxip = target_cpu
        .strip_prefix("gfx")
        .unwrap_or_else(|| panic!("target-cpu '{target_cpu}' did not start with gfx"));

    println!("cargo::rustc-link-arg={}/ockl.bc", device_libs);
    println!(
        "cargo::rustc-link-arg={}/oclc_isa_version_{}.bc",
        device_libs, gfxip,
    );
    println!(
        "cargo::rustc-link-arg={}/oclc_abi_version_500.bc",
        device_libs,
    );

    let mut is_wave64 = target_features.contains("wavefrontsize64");
    // wave64 is the default on gfx9 and before
    is_wave64 |= gfxip.starts_with('9') && gfxip.len() == 3;
    is_wave64 |= gfxip.starts_with("9-") && gfxip.ends_with("-generic");
    let is_wave64 = is_wave64;
    let wavesize = if is_wave64 { 64 } else { 32 };
    println!(
        "cargo::rustc-link-arg={}/oclc_wavefrontsize64_{}.bc",
        device_libs,
        if is_wave64 { "on" } else { "off" },
    );

    println!("cargo::rerun-if-changed={}/util{}.bc", cur_dir, wavesize);
    println!("cargo::rustc-link-arg={}/util{}.bc", cur_dir, wavesize);

    // Workarounds to make linker-plugin-lto work
    println!("cargo::rustc-link-arg=--undefined-version");
    println!("cargo::rustc-link-arg=--no-gc-sections");
}
