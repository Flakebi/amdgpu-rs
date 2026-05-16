//! GPU/shared code
#![cfg_attr(feature = "gpu", no_std)]
#![cfg_attr(feature = "gpu", feature(abi_gpu_kernel))]
#[cfg(feature = "gpu")]
extern crate alloc;

use gpu_kernel::kernel;

#[cfg(feature = "gpu")]
use amdgpu_device_libs::prelude::*;

// TODO Do nothing for GPUs
#[cfg(not(feature = "gpu"))]
gpu_kernel::kernel_lib!();

#[kernel]
pub fn test(i: i32) {
    println!("Hello World from the GPU!");
    // i + 42
}
