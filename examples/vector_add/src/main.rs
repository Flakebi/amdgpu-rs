#![cfg_attr(feature = "gpu", no_std)]
#![cfg_attr(feature = "gpu", feature(abi_gpu_kernel))]

use gpu_kernel::kernel;

gpu_kernel::kernel_lib!();

#[kernel]
pub fn test(i: i32) {
    use gpu_kernel::prelude::*;

    println!("Hello World from the GPU!");
}

#[cfg(not(feature = "gpu"))]
fn main() {
    use gpu_kernel::LaunchConfig;

    println!("Hello, world!");

    test(
        LaunchConfig::new()
            .threads_per_workgroup([4, 1, 1])
            .workgroups([1, 1, 1])
            .build(),
        1,
    );
}

/*fn gpu_with_std_same_crate() {
    gpu_launch(LaunchConfig::new(…), || println!("hi!"));
    gpu_launch(LaunchConfig::new(…), || println!("ho!"));
}

// Implementation

// With mangling
#[kernel]
fn launch_kernel<F: FnOnce()>(f: F) {
    f();
}

fn gpu_launch<F: FnOnce()>(config: LaunchConfig, f: F) {
    launch_kernel<F>(config, f);
}*/
