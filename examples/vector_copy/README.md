# Vector Copy Example

A simple GPU kernel where every thread copies an element from an input vector to an output vector.

The crate has no dependencies apart from `core`.

## Compile

The concrete hardware architecture needs to be set through `CARGO_BUILD_RUSTFLAGS`:
```bash
$ CARGO_BUILD_RUSTFLAGS='-Ctarget-cpu=gfx900' cargo build --release
```

Replace `gfx900` above with the version on your system.
The gfx version is printed e.g. by `rocminfo` or the [`default-cpu`](../default-cpu) program.

## Running

Compile this crate (preferably in release mode, debug mode tends to be very slow),
then pass the compiled kernel (`target/amdgcn-amd-amdhsa/release/vector_copy.elf`) as an argument to [`default-cpu`](../default-cpu).

The expected output is (among other information)
```bash
$ cargo run ../vector_copy/target/amdgcn-amd-amdhsa/release/vector_copy.elf
…
Failed for the second half! (This is expected for the simple vector_copy)
…
```
