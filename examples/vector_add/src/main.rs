use gpu_kernel::LaunchConfig;

use vector_add::*;

fn main() {
    println!("Hello, world!");

    test(LaunchConfig::threads(32), 1);
}
