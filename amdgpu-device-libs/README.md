# amdgpu-device-libs [![docs.rs](https://docs.rs/amdgpu-device-libs/badge.svg)](https://docs.rs/amdgpu-device-libs)

Support library for the amdgpu target.

By default, the amdgpu target supports `core`, but not `std`.
`alloc` is supported when a global allocator is specified.

`amdgpu-device-libs` brings some std-like features to the amdgpu target:

- `print!()` and `println!()` macros for printing on the host stdout
- A global allocator to support `alloc`
- A panic handler
- Access to more intrinsics and device-libs functions

All these features are enabled by default, but can be turned on selectively with `default-features = false, features = […]`.

`amdgpu-device-libs` works by linking to the [ROCm device-libs](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs) and a pre-compiled helper library.
The libraries are linked from a ROCm installation.
To make sure the libraries are found, set the environment variable `ROCM_PATH` or `ROCM_DEVICE_LIB_PATH` (higher priority if it is set).
It looks for `amdgcn/bitcode/*.bc` files in this path.

### Usage

Create a new cargo library project and change it to compile a cdylib:
```toml
# Cargo.toml
# Force lto
[profile.dev]
lto = true
[profile.release]
lto = true

[lib]
# Compile a cdylib
crate-type = ["cdylib"]

[build-dependencies]
# Used in build script to specify linker flags and link in device-libs
amdgpu-device-libs-build = { path = "../../amdgpu-device-libs-build" }

[dependencies]
amdgpu-device-libs = { path = "../../amdgpu-device-libs" }
```

Add extra flags in `.cargo/config.toml`:
```toml
# .cargo/config.toml
[build]
target = "amdgcn-amd-amdhsa"
# Enable linker-plugin-lto and workarounds
# Either add -Ctarget-cpu=gfx<version> here or specify it in CARGO_BUILD_RUSTFLAGS='-Ctarget-cpu=gfx<version>'
rustflags = ["-Clinker-plugin-lto", "-Zemit-thin-lto=no"]

[unstable]
build-std = ["core", "alloc"]
```

And add a `build.rs` build script that links to the required libraries:
```rust
// build.rs
fn main() {
    amdgpu_device_libs_build::build();
}
```

### Example

Minimal usage sample, see [`examples/println`](https://github.com/Flakebi/amdgpu-rs/tree/main/examples/println) for the full code.
```rust
#![feature(abi_gpu_kernel)]
#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use amdgpu_device_libs::prelude::*;

#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn kernel(output: *mut u32) {
    let wg_id = workgroup_id_x();
    let id = workitem_id_x();
    let dispatch = dispatch_ptr();
    let complete_id = wg_id as usize * dispatch.workgroup_size_x as usize + id as usize;

    println!("Hello world from the GPU! (thread {wg_id}-{id})");

    let mut v = Vec::<u32>::new();
    for i in 0..100 {
        v.push(100 + i);
    }

    unsafe {
        *output.add(complete_id) = v[complete_id];
    }
}
```
