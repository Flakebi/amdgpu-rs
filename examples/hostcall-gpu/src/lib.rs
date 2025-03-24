#![feature(abi_gpu_kernel)]
#![no_std]

extern crate alloc;

use amdgpu_device_libs::prelude::*;

#[unsafe(no_mangle)]
pub extern "gpu-kernel" fn kernel(host_func: u64) {
    // Get workgroup and thread id
    let wg_id = workgroup_id_x();
    let id = workitem_id_x();

    // First thread in every workgroups does a hostcall, passing the workgroup id
    if id == 0 {
        unsafe {
            println!("Calling host function {host_func:#x} with argument {wg_id}");
            let res =
                amdgpu_device_libs::call_host_function(host_func, wg_id.into(), 0, 0, 0, 0, 0, 0);
            println!("Got result from host: {res}");
        }
    }
}
