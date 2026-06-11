//! Similar to the vector_add example, but copies a larger vector and has more of a focus on performance.
//! Mostly to show off/test performance features.
#![cfg_attr(feature = "gpu", no_std)]
#![cfg_attr(feature = "gpu", feature(abi_gpu_kernel))]
#![cfg_attr(not(feature = "gpu"), feature(clone_from_ref))]

use gpu_kernel::kernel;

gpu_kernel::kernel_lib!();

/// This kernel adds numbers from two slices and writes the result into a third.
#[kernel]
#[allow(improper_ctypes_definitions, improper_gpu_kernel_arg)]
pub unsafe fn kernel(a: &[u32], b: &[u32], c: *mut u32) {
    use gpu_kernel::intrinsics::{workgroup_id_x, workitem_id_x};

    // Compute own, global id
    let id = workitem_id_x() as usize + 32 * workgroup_id_x() as usize;

    // Add multiple input numbers and store into output
    let mut sum = 0;
    for i in 0..100 {
        // This accesses memory in  multiple places to demonstrate a performance difference
        // when the memory is in CPU or GPU memory
        sum += a[id] + b[id] + a[(id * i) % a.len()] - b[a.len().wrapping_sub(id * i) % a.len()];
    }
    unsafe {
        *c.add(id) = sum;
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {
    use gpu_kernel::{GpuAlloc, GpuBox, LaunchConfig};

    // Create two vectors a and b to add together
    let mut a = Vec::new();
    let mut b = Vec::new();
    // Use a multiple of 32, so that the kernel does not need to handle boundaries
    for i in 0..3200_000 {
        a.push(i);
        b.push(i);
    }

    // Copy vectors to GPU
    let a = GpuBox::clone_from_ref_in(a.as_slice(), GpuAlloc);
    let b = GpuBox::clone_from_ref_in(b.as_slice(), GpuAlloc);

    // Create vector c to hold the results.
    // Create it on the CPU as that seems to be faster than on the GPU
    let mut c_gpu = Box::new_uninit_slice(a.len());
    // Creating it on the GPU would look like this:
    // let mut c_gpu = GpuBox::new_uninit_slice_in(a.len(), GpuAlloc);

    unsafe {
        kernel.launch(
            LaunchConfig::new()
                .threads_per_workgroup([32, 1, 1])
                .workgroups([(a.len() / 32) as u32, 1, 1]),
            &a,
            &b,
            c_gpu.as_mut_ptr() as *mut _,
        );
    }

    let c_gpu = unsafe { c_gpu.assume_init() };

    // Copy c back to CPU
    let mut c = [0u32; 100];
    c.copy_from_slice(&c_gpu[..100]);

    println!("Result: {:?}", c);
}
