# CPU Part

The CPU binary that launches various example kernels on the GPU.
It uses HIP and is loosely based on the [ROCm/rocm-examples/module_api](https://github.com/ROCm/rocm-examples/blob/31e7ba9bb2e1a401b90805d7b242b860ac249623/HIP-Basic/module_api) example.

The [`hip-runtime-sys`](https://github.com/cjordan/hip-sys) crate is used and expects `ROCM_PATH` to point to an installation of ROCm.

## Usage

E.g. after compiling the [`vector_copy`](../vector_copy) example with `CARGO_BUILD_RUSTFLAGS='-Ctarget-cpu=gfx1036' cargo build --release` (replace `gfx<num>` with the version from your system).
```bash
$ cargo run ../vector_copy/target/amdgcn-amd-amdhsa/release/vector_copy.elf
Driver Version: 60032831
Runtime Version: 60032831
Device Count: 2
Device 0
  AMD Radeon Graphics (gfx1036) | multi 0
  mem    | VRAM: 30GiB, shared/block: 64KiB, 
  thread | max/block: 1024, warpSize 32, 1 processors with 2048 max threads, max [1024 1024 1024]
  grid   | max [2147483647 65536 65536]
Set device 0
Alloc memory
Copy memory from 0x55ee663b1c40 (cpu) to 0x1141400000 (gpu)
Copy memory from 0x55ee663b2050 (cpu) to 0x1141401000 (gpu)
Load module from ../vector_copy/target/amdgcn-amd-amdhsa/release/vector_copy.elf
Get function kernel
Launch kernel
Wait for finish
Copy memory back
Output: [00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 0a, 0b, 0c, 0d, 0e, 0f, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1a, 1b, 1c, 1d, 1e, 1f]
Failed for the second half! (This is expected for the simple vector_copy)
Free
Finished
```

When multiple devices are available, by default device `0` is picked.
A different one can be specified with e.g. `-d 1`.

## Debugging

Run with `AMD_LOG_LEVEL=7` (or lower) to get debug output.

## Cross Compiling

It is possible to cross-compile a Windows Rust HIP application from Linux.

For cross-compilation, it is necessary to get a copy of the `amdhip64` lib and dll.
They can be obtained on Linux, the process is roughly like this:
Download the SDK from https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html, then run the installer in wine and copy the extracted libraries.
```bash
# Create a new wine prefix
mkdir wine
cd wine
export WINEPREFIX=$(pwd)
# Running the package installer will fail, but that is ok, it extracts the needed installers before it fails
wine ../AMD-Software-PRO-Edition-24.Q4-Win10-Win11-For-HIP.exe
# Run core SDK installer
wine drive_c/AMD/AMD-Software-Installer/Packages/Apps/ROCmSDKPackages/SDKCore/ROCm_SDK_Core.msi
# Copy extracted libraries to Rust project
cp 'drive_c/Program Files/AMD/ROCm/6.2/lib/amdhip64.lib' 'drive_c/Program Files/AMD/ROCm/6.2/bin/amdhip64_6.dll' /path/to/default-cpu/
# Build Rust project for Windows
cargo build --target x86_64-pc-windows-gnu --release
```
