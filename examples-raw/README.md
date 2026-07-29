# amdgpu-rs Examples

Examples for using the Rust amdgpu target and `amdgpu-device-libs`.

The examples use the [`hip-runtime-sys`](https://github.com/cjordan/hip-sys) crate on the CPU side to launch kernels.
This expects `HIP_PATH` to point to an installation of ROCm.

To compile examples, set the concrete hardware architecture with `CARGO_BUILD_RUSTFLAGS='-Ctarget-cpu=gfx900'` (replace `gfx900` with the name printed by `rocminfo` on the used system).

A basic example for a GPU kernel is in [`vector_copy`](./vector_copy).
This does not use the `amdgpu-device-libs` library, it only depends on `core`.

The GPU kernels need accompanying CPU code to launch them.
The CPU code to launch the example kernels is in [`default-cpu`](./default-cpu) (unless explicitly specified otherwise).

A simple example that uses `amdgpu-device-libs` is [`println`](./println).
