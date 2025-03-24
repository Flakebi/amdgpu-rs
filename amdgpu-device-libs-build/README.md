# amdgpu-device-libs-build [![docs.rs](https://docs.rs/amdgpu-device-libs-build/badge.svg)](https://docs.rs/amdgpu-device-libs-build)

Build script support for `amdgpu-device-libs`.

Adds linker flags to link device-libs.
Add `amdgpu-device-libs-build` as a `build-dependency` and call it in the build script.
```rust
// build.rs
fn main() {
    amdgpu_device_libs_build::build();
}
```

This link to the [ROCm device-libs](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs) and a pre-compiled helper library.
The libraries are linked from a ROCm installation.
To make sure the libraries are found, set the environment variable `ROCM_PATH` or `ROCM_DEVICE_LIB_PATH` (higher priority if it is set).
It looks for `amdgcn/bitcode/*.bc` files in this path.
See the documentation of [`amdgpu-device-libs`](https://docs.rs/amdgpu-device-libs) for more information.

## Build utility library

The code to emit device code for `printf` is in clang, therefore it cannot be used easily from rustc or any Rust library.
We workaround this by compiling a simple hip file ([`util.hip`](./util.hip)) with clang and that exposes a print function.
`util.hip` is compiled to LLVM bitcode, we then massage this bitcode, stripping unnecessary parts to make it generic over the gfx hardware architecture.
The bitcode files resulting from this ship with this library as bitcode binaries.
They are fairly small (can be inspected by hand) and not needing a hip-enabled clang makes it much easier to compile Rust programs for amdgpu.
The `compile-util.sh` script is used to create the bitcode.
It needs `ROCM_DEVICE_LIB_PATH` pointing to a path that contains `amdgcn/bitcode/*.bc` files.
