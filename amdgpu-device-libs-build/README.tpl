# {{crate}} [![docs.rs](https://docs.rs/amdgpu-device-libs-build/badge.svg)](https://docs.rs/amdgpu-device-libs-build)

{{readme}}

## Build utility library

The code to emit device code for `printf` is in clang, therefore it cannot be used easily from rustc or any Rust library.
We workaround this by compiling a simple hip file ([`util.hip`](./util.hip)) with clang and that exposes a print function.
`util.hip` is compiled to LLVM bitcode, we then massage this bitcode, stripping unnecessary parts to make it generic over the gfx hardware architecture.
The bitcode files resulting from this ship with this library as bitcode binaries.
They are fairly small (can be inspected by hand) and not needing a hip-enabled clang makes it much easier to compile Rust programs for amdgpu.
The `compile-util.sh` script is used to create the bitcode.
It needs `ROCM_DEVICE_LIB_PATH` pointing to a path that contains `amdgcn/bitcode/*.bc` files.
