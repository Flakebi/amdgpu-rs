#![cfg_attr(feature = "gpu", no_std)]
#![cfg_attr(feature = "gpu", feature(abi_gpu_kernel))]

use gpu_kernel::kernel;

gpu_kernel::kernel_lib!();

#[kernel]
#[allow(improper_ctypes_definitions, improper_gpu_kernel_arg)]
pub fn test<'a, 'b>(a: &'a [u32], b: &'b [u32], c: *mut u32) {
    use gpu_kernel::prelude::*;

    let id = workitem_id_x() as usize;
    println!("Hello World #{} from the GPU!", id);

    unsafe {
        *c.offset(id as isize) = a[id] + b[id];
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {
    use gpu_kernel::LaunchConfig;

    println!("Hello, world!");

    // TODO Use global hip allocator for unified mem?

    let mut a = Vec::new();
    let mut b = Vec::new();
    for i in 0..10 {
        a.push(i);
        b.push(i);
    }
    let mut c = Vec::new();
    c.resize(a.len(), 0);

    test(
        LaunchConfig::new()
            .threads_per_workgroup([a.len() as u32, 1, 1])
            .workgroups([1, 1, 1])
            .build(),
        &a,
        &b,
        c.as_mut_ptr(),
    );

    println!("Result: {c:?}");
}
