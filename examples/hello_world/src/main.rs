#![cfg_attr(feature = "gpu", no_std, feature(abi_gpu_kernel))]

use gpu_kernel::kernel;

gpu_kernel::kernel_lib!();

#[kernel]
fn kernel(s: &str) {
    use gpu_kernel::intrinsics::workitem_id_x;

    let id = workitem_id_x();
    println!("Hello {s} from #{}!", id);
}

#[cfg(not(feature = "gpu"))]
fn main() {
    use gpu_kernel::LaunchConfig;

    let s = "World".to_string();

    kernel.launch(
        LaunchConfig::new()
            .threads_per_workgroup([10, 1, 1])
            .workgroups([1, 1, 1]),
        &s,
    );
}
