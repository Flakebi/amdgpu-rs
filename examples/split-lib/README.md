# Split GPU and CPU code

While writing GPU kernels and CPU code in the same file looks appealing, it forces the whole crate to be compilable with `no_std`.
In most examples this is achieved by guarding `main` with `#[cfg(not(feature = "gpu"))]` but that is impractical for larger programs.
A solution that always works is splitting a program into a no-std library crate containing GPU kernels and a binary crate that depends on the library.

Fortunately, we can apply a trick to create two crates inside one: The `lib.rs` can be no-std compatible and contain GPU kernels, while the `main.rs` can contain CPU code and use std.
The `main.rs` depends on/imports the lib.

To make this work with dependencies, shared crates are added to the `[dependencies]` section in `Cargo.toml`, while CPU-only dependencies are added to target-specific dependencies that do not need to be compiled for the GPU:
```toml
# Dependencies shared between GPU and CPU code, must be no-std
[dependencies]
gpu-kernel = "…"

# Dependencies for the CPU code, can use std
[target.'cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))'.dependencies]
# Add CPU-only dependencies here
```
