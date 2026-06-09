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
