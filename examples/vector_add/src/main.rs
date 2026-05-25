#![cfg_attr(feature = "gpu", no_std)]
#![cfg_attr(feature = "gpu", feature(abi_gpu_kernel))]

use gpu_kernel::kernel;

gpu_kernel::kernel_lib!();

#[kernel]
pub fn test(a: &[u32], b: &[u32]) {
    use gpu_kernel::prelude::*;

    println!("Hello World #{} from the GPU!", a[0]);
}

#[cfg(not(feature = "gpu"))]
fn main() {
    use gpu_kernel::LaunchConfig;

    println!("Hello, world!");

    // TODO Implement actual vector-add
    // TODO Use global hip allocator for shared mem and pass a slice?

    let mut a = Vec::new();
    let mut b = Vec::new();
    for i in 0..10 {
        a.push(i);
        b.push(i);
    }

    test(
        LaunchConfig::new()
            .threads_per_workgroup([4, 1, 1])
            .workgroups([1, 1, 1])
            .build(),
        &a,
        &b,
    );
}
