use gpu_kernel::LaunchConfig;

use vector_add::*;

fn main() {
    println!("Hello, world!");

    test(LaunchConfig::threads(32), 1);
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
