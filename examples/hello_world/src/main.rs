#![cfg_attr(feature = "gpu", no_std)]
#![cfg_attr(feature = "gpu", feature(abi_gpu_kernel))]

use gpu_kernel::kernel;

gpu_kernel::kernel_lib!();

#[kernel]
#[allow(improper_ctypes_definitions, improper_gpu_kernel_arg)]
pub unsafe fn kernel(s: &str) {
    use gpu_kernel::intrinsics::workitem_id_x;

    let id = workitem_id_x() as usize;
    println!("Hello {s} from #{}!", id);
}

#[cfg(not(feature = "gpu"))]
fn main() {
    use gpu_kernel::LaunchConfig;

    // Only heap variables are shared on dedicated AMD GPUs,
    // so create the string on the heap
    let s = "World".to_string();

    unsafe {
        kernel(
            LaunchConfig::new()
                .threads_per_workgroup([10, 1, 1])
                .workgroups([1, 1, 1])
                .build(),
            &s,
        );
    }
}
