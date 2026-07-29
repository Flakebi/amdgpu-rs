# gpu-kernel [![docs.rs](https://docs.rs/gpu-kernel/badge.svg)](https://docs.rs/gpu-kernel)

Running Rust code on a GPU is not as hard as it might sound and here is how it’s done!

Let us start with the code, it takes just a few lines:
```rust
// main.rs
// GPU code is no-std
#![cfg_attr(feature = "gpu", no_std, feature(abi_gpu_kernel))]

// Macro to compile and include the GPU code
gpu_kernel::kernel_lib!();

// Define a kernel, this function runs on the GPU
#[gpu_kernel::kernel]
fn kernel(s: &str) {
    let id = gpu_kernel::intrinsics::workitem_id_x();
    println!("Hello {s} from thread #{}!", id);
}

#[cfg(not(feature = "gpu"))]
fn main() {
    let s = "World".to_string();
    // Launch 10 threads on the GPU
    kernel.launch(
        gpu_kernel::LaunchConfig::new()
            .threads_per_workgroup([10, 1, 1])
            .workgroups([1, 1, 1]),
        &s,
    );
}
```

This is all Rust code, it prints hello world from the GPU for each started thread:
```bash
$ cargo run
Hello World from thread #0!
Hello World from thread #1!
Hello World from thread #2!
Hello World from thread #3!
Hello World from thread #4!
Hello World from thread #5!
Hello World from thread #6!
Hello World from thread #7!
Hello World from thread #8!
Hello World from thread #9!
```

In `Cargo.toml`, we add `gpu-kernel` as a dependency:
```toml
# Cargo.toml
[package]
name = "hello_world"
version = "0.1.0"
edition = "2024"

# This gets defined when building for the gpu.
# It can be omitted when using target_arch or similar for cfg conditions, it exists for convenience only.
[features]
gpu = []

[dependencies]
gpu-kernel = "0.1"
```

For `cargo run` to work, the GPU compute runtime needs to be installed, see the next section.

### Setup

Currently, AMD GPUs are supported.
Contributions for other Rust GPU backends are welcome, adding support to `gpu-kernel` should be relatively straightforward.

1. Install ROCm, on Ubuntu 26.04, this is a simple `apt install rocm-dev`
1. Add `rust-src` to rustup to support build-std: `rustup component add rust-src`
1. Configure your GPU in cargo’s config, find your version with `rocminfo | grep gfx`:
   ```toml
   # ~/.cargo/config.toml
   [target.amdgcn-amd-amdhsa]
   rustflags = ["-Ctarget-cpu=gfx<your version>"]
   # If rocminfo shows xnack- for your GPU, add "-Ctarget-feature=-xnack-support"
   ```
   Alternatively, specify the version through an environment variable: `CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS=-Ctarget-cpu=gfx<your version>`
1. Set `HIP_PATH=/usr` for `hip-runtime-sys` to find the hip headers

On NixOS, skip step 4 and add `rocmPackages.clr` to your dev shell to automagically set `HIP_DEVICE_LIB_PATH` and `HIP_PATH` or alternatively set `HIP_DEVICE_LIB_PATH="${rocmPackages.rocm-device-libs}/amdgcn/bitcode"` and `HIP_PATH="${rocmPackages.clr}"`.

### Settings

Configuration files like `.cargo/config.toml` and `~/.cargo/config.toml` can be used to specify compiler flags as described in the [setup](#setup) section.

Additionally, a few of environment variables can be set:

| Env variable                               | Default                                         | Example               | Description                                                              |
|--------------------------------------------|-------------------------------------------------|-----------------------|--------------------------------------------------------------------------|
| `HIP_PATH`                                 | `/opt/rocm/hip`                                 | `/usr`                | Path to the hip installation to find headers                             |
| `HIP_DEVICE_LIB_PATH`                      | `$(hipconfig -l)/../lib/clang/*/amdgcn/bitcode` |                       | Path to device libs, ends with `amdgcn/bitcode` and contains `.bc` files |
| `CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS` | empty                                           | `-Ctarget-cpu=gfx900` | RUSTFLAGS used to compile amdgpu GPU code                                |
| `CARGO_TARGET_AMDGCN_AMD_AMDHSA_FLAGS`     | empty                                           | `-v`                  | Cargo flags used to compile amdgpu GPU code                              |

Several flags are added automatically to the GPU compilation.

- If a `gpu` feature is defined in `Cargo.toml`, `--features=gpu` is passed to cargo
- The `crate-type` is set to `cdylib`
- Device libs are added to `link-arg`s and `-Clinker-plugin-lto`
- core and alloc are built with `-Zbuild-std=core,alloc`
- In debug mode, `opt-level=2` is set, as no optimizations can lead to crashes or compilation failures in the backend
- In release mode, `panic=immediate-abort` is set for performance, so no panic messages are available
