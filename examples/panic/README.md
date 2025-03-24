# `panic!` Example

A GPU kernel that panics on the GPU.

The crate depends on `amdgpu-device-libs`.

## Compile

The concrete hardware architecture needs to be set through `CARGO_BUILD_RUSTFLAGS`:
```bash
$ CARGO_BUILD_RUSTFLAGS='-Ctarget-cpu=gfx900' cargo build --release
```

Replace `gfx900` above with the version on your system.
The gfx version is printed e.g. by `rocminfo` or the [`default-cpu`](../default-cpu) program.

## Running

Compile this crate (preferably in release mode, debug mode tends to be very slow),
then pass the compiled kernel (`target/amdgcn-amd-amdhsa/release/panic.elf`) as an argument to [`default-cpu`](../default-cpu).

The expected output is (among other information)
```bash
$ cargo run ../panic/target/amdgcn-amd-amdhsa/release/panic.elf
…
workgroup 0,0,0 thread 0,0,0 panicked at src/lib.rs:12:5:
assertion `left != right` failed: Expected workitem id to be non-0 (this will panic)
  left: 0
 right: 0
workgroup 1,0,0 thread 0,0,0 panicked at src/lib.rs:12:5:
assertion `left != right` failed: Expected workitem id to be non-0 (this will panic)
  left: 0
 right: 0
```
