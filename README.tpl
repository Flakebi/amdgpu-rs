# amdgpu-rs

Examples and support libraries for the amdgpu Rust target.

## amdgpu-device-libs [![docs.rs](https://docs.rs/amdgpu-device-libs/badge.svg)](https://docs.rs/amdgpu-device-libs)

{{readme}}

## Examples

The examples use the [`hip-runtime-sys`](https://github.com/cjordan/hip-sys) crate on the CPU side to launch kernels.
This expects `ROCM_PATH` to point to an installation of ROCm.

To compile examples, set the concrete hardware architecture with `CARGO_BUILD_RUSTFLAGS='-Ctarget-cpu=gfx900'` (replace `gfx900` with the name printed by `rocminfo` on the used system).

See [`examples`](./examples).

## License

Licensed under either of

 * [Apache License, Version 2.0](LICENSE-APACHE)
 * [MIT license](LICENSE-MIT)

at your option.
