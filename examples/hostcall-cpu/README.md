# Hostcall CPU Part

The CPU binary that exposes a hostcall function and launches the kernel from hostcall-gpu.

The GPU kernel calls `host_func` on the CPU.

## Usage

After compiling the [`hostcall-gpu`](../hostcall-gpu) example with `CARGO_BUILD_RUSTFLAGS='-Ctarget-cpu=gfx900' cargo build --release` (replace `gfx<num>` with the version from your system).
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

When multiple devices are available, by default device `0` is picked.
A different one can be specified with e.g. `-d 1`.
