# Hostcall Example

A GPU kernel that uses hostcalls to call a function on the CPU from the GPU.

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
then pass the compiled kernel (`target/amdgcn-amd-amdhsa/release/hostcall-gpu.elf`) as an argument to [`hostcall-cpu`](../hostcall-cpu).

The expected output is (among other information)
```bash
$ cargo run ../hostcall-gpu/target/amdgcn-amd-amdhsa/release/hostcall_gpu.elf
…
Load module from ../hostcall-gpu/target/amdgcn-amd-amdhsa/release/hostcall_gpu.elf
Get function kernel
host_func has address 0x55d25e20ff40
Launch kernel
Wait for finish
Calling host function 0x55d25e20ff40 with argument 1
Calling host function 0x55d25e20ff40 with argument 0
Function on the host, got 0
Function on the host, got 1
Got result from host: 42
Got result from host: 43
Free
Finished
```
