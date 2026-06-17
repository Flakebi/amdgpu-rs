// GPU code is no-std
#![cfg_attr(feature = "gpu", no_std, feature(abi_gpu_kernel))]

// Macro to compile and include the GPU code
gpu_kernel::kernel_lib!();

// Define a kernel, this function runs on the GPU
#[gpu_kernel::kernel]
fn kernel(s: &str) {
    let id = gpu_kernel::intrinsics::workitem_id_x();
    println!("Hello {s} from thread #{}!", id);
}

#[cfg(not(feature = "gpu"))]
fn main() {
    let s = "World".to_string();
    // Launch 10 threads on the GPU
    kernel.launch(
        gpu_kernel::LaunchConfig::new()
            .threads_per_workgroup([10, 1, 1])
            .workgroups([1, 1, 1]),
        &s,
    );
}
