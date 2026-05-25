#![cfg_attr(any(target_arch = "amdgpu", target_arch = "nvptx64"), no_std)]

pub use gpu_kernel_proc_macros::{kernel, kernel_lib_impl};

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
#[doc(hidden)]
pub use hip_runtime_sys;

#[cfg(target_arch = "amdgpu")]
pub use amdgpu_device_libs::*;

#[macro_export]
macro_rules! kernel_lib {
    () => {
        #[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
        ::gpu_kernel::kernel_lib_impl!();

        #[cfg(any(target_arch = "amdgpu", target_arch = "nvptx64"))]
        extern crate alloc;
    };
}

pub struct LaunchConfig {
    pub workgroups: [u32; 3],
    pub threads_per_workgroups: [u32; 3],
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
pub struct Kernel {
    func: hip_runtime_sys::hipFunction_t,
}
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl Send for Kernel {}
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl Sync for Kernel {}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
pub struct Module(pub hip_runtime_sys::hipModule_t);
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl Send for Module {}
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl Sync for Module {}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
impl Kernel {
    #[doc(hidden)]
    pub fn new(func: hip_runtime_sys::hipFunction_t) -> Self {
        Self { func }
    }

    #[doc(hidden)]
    pub fn launch<T: 'static>(&self, launch_config: LaunchConfig, mut args: T) {
        use std::ffi;

        let mut size = std::mem::size_of::<T>();
        let args = &mut args;

        #[allow(clippy::manual_dangling_ptr)]
        let mut config = [
            0x1 as *mut ffi::c_void,                          // Next come arguments
            args as *mut _ as *mut ffi::c_void,               // Pointer to arguments
            0x2 as *mut ffi::c_void,                          // Next comes size
            std::ptr::addr_of_mut!(size) as *mut ffi::c_void, // Pointer to size of arguments
            0x3 as *mut ffi::c_void,                          // End
        ];

        unsafe {
            // Launch two workgroups (2x1x1), each of the size (LEN/2)x1x1
            let result = hip_runtime_sys::hipModuleLaunchKernel(
                self.func,
                launch_config.workgroups[0],
                launch_config.workgroups[1],
                launch_config.workgroups[2],
                launch_config.threads_per_workgroups[0],
                launch_config.threads_per_workgroups[1],
                launch_config.threads_per_workgroups[2],
                0,                    // sharedMemBytes for extern shared variables
                std::ptr::null_mut(), // stream
                std::ptr::null_mut(), // params (unimplemented in hip)
                config.as_mut_ptr(),  // arguments
            );
            assert_eq!(
                result,
                hip_runtime_sys::hipError_t::hipSuccess,
                "Failed to launch kernel"
            );

            let result = hip_runtime_sys::hipDeviceSynchronize();
            assert_eq!(
                result,
                hip_runtime_sys::hipError_t::hipSuccess,
                "Failed to wait for kernel to finish"
            );
        }
    }
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
