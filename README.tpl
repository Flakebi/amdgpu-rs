# Rust on (AMD) GPUs [![docs.rs](https://docs.rs/gpu-kernel/badge.svg)](https://docs.rs/gpu-kernel)

{{readme}}

## Examples

More examples can be found in [`examples`](./examples)

## amdgpu-device-libs [![docs.rs](https://docs.rs/amdgpu-device-libs/badge.svg)](https://docs.rs/amdgpu-device-libs)

This repo also contains support libraries for the amdgpu Rust target and more low-level examples using these.

See the [`amdgpu-device-libs`](./amdgpu-device-libs) folder for docs and [`examples-amdgpu-raw`](./examples-amdgpu-raw) for examples.

## Tests

All examples can be run with `nix run .#runExamples`.
To specify a non-default device, e.g. 1, use `ROCR_VISIBLE_DEVICES=1 nix run .#runExamples`.

## License

Licensed under either of

 * [Apache License, Version 2.0](LICENSE-APACHE)
 * [MIT license](LICENSE-MIT)

at your option.
