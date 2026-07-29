#![cfg_attr(feature = "gpu", no_std, feature(abi_gpu_kernel))]

use gpu_kernel::{ThreadIndexedSlice, kernel};

gpu_kernel::kernel_lib!();

/// This kernel adds numbers from two slices and writes the result into a third.
#[kernel]
fn kernel(a: &[u32], b: &[u32], mut c: ThreadIndexedSlice<'_, u32>) {
    use gpu_kernel::intrinsics::workitem_id_x;

    let id = workitem_id_x() as usize;

    // Add two input numbers and store into output
    *c.get_mut() = a[id] + b[id];
}

#[cfg(not(feature = "gpu"))]
fn main() {
    use gpu_kernel::LaunchConfig;

    // TODO Move to amdgpu-rs

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
    // See the vector_add_fast example for how to improve this.
    // Only heap variables are shared on dedicated AMD GPUs, so this cannot be constant slices.

    kernel.launch(
        LaunchConfig::new()
            .threads_per_workgroup([a.len() as u32, 1, 1])
            .workgroups([1, 1, 1]),
        &a,
        &b,
        &mut c,
    );

    println!("Result: {c:?}");
}
