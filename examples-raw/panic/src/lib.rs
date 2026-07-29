#![feature(abi_gpu_kernel)]
#![no_std]

use amdgpu_device_libs::prelude::*;

// Take two arguments, so it can be launched by default-cpu.
#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn kernel(_: *const u8, _: *mut u8) {
    let id = workitem_id_x();

    // Let the first thread in every workgroup panic
    assert_ne!(id, 0, "Expected workitem id to be non-0 (this will panic)");
}
