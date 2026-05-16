#![no_std]

pub use gpu_kernel_proc_macros::{kernel, kernel_lib};

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
#[doc(hidden)]
pub use hip_runtime_sys;

pub struct LaunchConfig {
    pub workgroups: [u32; 3],
    pub threads_per_workgroups: [u32; 3],
}

pub struct Kernel {
    // TODO, make constructor function instead
    pub id: u32,
}

impl LaunchConfig {
    pub fn threads(num: u32) -> Self {
        let default_wg_size = 32;
        Self {
            // TODO This will activate more threads than intended...
            workgroups: [num.div_ceil(default_wg_size), 1, 1],
            threads_per_workgroups: [default_wg_size, 1, 1],
        }
    }
}
