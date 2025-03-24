# `println!` Example

A GPU kernel that uses `println!` from the GPU.
It also implements `vector_copy` with multiple workgroups where every thread copies an element from an input vector to an output vector.

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
then pass the compiled kernel (`target/amdgcn-amd-amdhsa/release/println.elf`) as an argument to [`default-cpu`](../default-cpu).

The expected output is (among other information)
```bash
$ cargo run ../println/target/amdgcn-amd-amdhsa/release/println.elf
…
Launch kernel
Wait for finish
Hello world from the GPU! (thread 0-0)
Hello world from the GPU! (thread 1-0)
Size 512x1x1, wavefrontsize: 32, group_segment_size: 512, groupstaticsize: 0
Size 512x1x1, wavefrontsize: 32, group_segment_size: 512, groupstaticsize: 0
Copy memory back
Output: [00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 0a, 0b, 0c, 0d, 0e, 0f, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1a, 1b, 1c, 1d, 1e, 1f]
PASSED!
…
```
