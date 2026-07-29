#![allow(internal_features)]
#![feature(abi_gpu_kernel, core_intrinsics, stdarch_amdgpu)]
#![no_std]

use core::arch::amdgpu;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::intrinsics::abort();
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kernel(input: *const u8, output: *mut u8) {
    let id = amdgpu::workitem_id_x() as usize;

    unsafe {
        // Copy input buffer to output buffer. Each invocation copies one byte.
        *output.add(id) = *input.add(id);
    }
}
