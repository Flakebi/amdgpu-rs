#![feature(abi_gpu_kernel)]
#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use amdgpu_device_libs::intrinsics;
use amdgpu_device_libs::prelude::*;

#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn kernel(input: *const u8, output: *mut u8) {
    // Get workgroup and thread id
    let wg_id = workgroup_id_x();
    let id = workitem_id_x();
    // Get workgroup size and compute complete id
    let dispatch = dispatch_ptr();
    let complete_id = wg_id as usize * dispatch.workgroup_size_x as usize + id as usize;

    // First thread in every workgroups prints some info
    if id == 0 {
        println!("Hello world from the GPU! (thread {wg_id}-{id})");
        println!(
            "Size {}x{}x{}, wavefrontsize: {}, group_segment_size: {}, groupstaticsize: {}",
            dispatch.workgroup_size_x,
            dispatch.workgroup_size_y,
            dispatch.workgroup_size_z,
            intrinsics::wavefrontsize(),
            dispatch.group_segment_size,   // Total size of shared memory
            intrinsics::groupstaticsize()  // Static size of shared memory
        );
    }

    // A dynamically allocated vector, just for demonstration that it works. It is not used for anything.
    let mut v = Vec::<u32>::new();
    for i in 0..100 {
        v.push(100 + i);
    }

    unsafe {
        *output.add(complete_id) = *input.add(complete_id);
    }
}
