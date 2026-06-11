#![cfg_attr(feature = "gpu", no_std)]
#![cfg_attr(feature = "gpu", feature(abi_gpu_kernel))]

use gpu_kernel::kernel;

gpu_kernel::kernel_lib!();

/// This kernel adds numbers from two slices and writes the result into a third.
#[kernel]
#[allow(improper_ctypes_definitions, improper_gpu_kernel_arg)]
pub unsafe fn kernel(a: &[u32], b: &[u32], c: *mut u32) {
    use gpu_kernel::intrinsics::workitem_id_x;

    let id = workitem_id_x() as usize;

    // Add two input numbers and store into output
    unsafe {
        *c.offset(id as isize) = a[id] + b[id];
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {
    use gpu_kernel::LaunchConfig;

    // TODO Move to amdgpu-rs
    // TODO Build with CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS='-Ctarget-cpu=gfx...' or set in ~/.cargo/config.toml
    // TODO Document everything and add readmes
    // TODO Test on Ubuntu podman
    // TODO any(doc) to cfgs

    // Create two vectors a and b to add together
    let mut a = Vec::new();
    let mut b = Vec::new();
    for i in 0..10 {
        a.push(i);
        b.push(i);
    }

    // Create vector c to hold the results
    let mut c = Vec::new();
    c.resize(a.len(), 0);

    // We pass CPU pointers to the kernel, which works fine, though is potentially slow.
    // hipMemoryAdvise can be used to improve this.
    // Only heap variables are shared on dedicated AMD GPUs, so this cannot be constant slices.

    unsafe {
        kernel.launch(
            LaunchConfig::new()
                .threads_per_workgroup([a.len() as u32, 1, 1])
                .workgroups([1, 1, 1]),
            &a,
            &b,
            c.as_mut_ptr(),
        );
    }

    println!("Result: {c:?}");
}
