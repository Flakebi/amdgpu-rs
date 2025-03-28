#![allow(internal_features)]
#![feature(abi_gpu_kernel, core_intrinsics, link_llvm_intrinsics)]
#![no_std]

unsafe extern "C" {
    #[link_name = "llvm.amdgcn.workitem.id.x"]
    pub safe fn workitem_id_x() -> u32;
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::intrinsics::abort();
}

#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kernel(input: *const u8, output: *mut u8) {
    let id = workitem_id_x() as usize;

    unsafe {
        // Copy input buffer to output buffer. Each invocation copies one byte.
        *output.add(id) = *input.add(id);
    }
}
