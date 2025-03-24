# amdgpu-rs Examples

Examples for using the Rust amdgpu target and `amdgpu-device-libs`.

A basic example for a GPU kernel is in [`vector_copy`](./vector_copy).
This does not use the `amdgpu-device-libs` library, it only depends on `core`.

The GPU kernels need accompanying CPU code to launch them.
The CPU code to launch the example kernels is in [`default-cpu`](./default-cpu) (unless explicitly specified otherwise).

A simple example that uses `amdgpu-device-libs` is [`println`](./println).
