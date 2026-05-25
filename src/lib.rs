#![cfg_attr(any(target_arch = "amdgpu", target_arch = "nvptx64"), no_std)]

pub use gpu_kernel_proc_macros::{kernel, kernel_lib_impl};

#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
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

#[non_exhaustive]
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct LaunchConfig {
    pub workgroups: [u32; 3],
    pub threads_per_workgroup: [u32; 3],
}

#[derive(Default, Clone, Eq, Hash, PartialEq)]
pub struct LaunchConfigBuilder {
    workgroups: Option<[u32; 3]>,
    threads_per_workgroup: Option<[u32; 3]>,
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
pub struct Module {
    #[cfg(feature = "amd")]
    module: hip_runtime_sys::hipModule_t,
}
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl Send for Module {}
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl Sync for Module {}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
pub struct Kernel {
    #[cfg(feature = "amd")]
    func: hip_runtime_sys::hipFunction_t,
}
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl Send for Kernel {}
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl Sync for Kernel {}

impl LaunchConfig {
    pub fn new() -> LaunchConfigBuilder {
        Default::default()
    }
}

impl LaunchConfigBuilder {
    pub fn workgroups(&mut self, workgroups: [u32; 3]) -> &mut Self {
        self.workgroups = Some(workgroups);
        self
    }

    pub fn threads_per_workgroup(&mut self, threads_per_workgroup: [u32; 3]) -> &mut Self {
        self.threads_per_workgroup = Some(threads_per_workgroup);
        self
    }

    /// Panics if a required field was not filled out
    pub fn build(&mut self) -> LaunchConfig {
        LaunchConfig {
            workgroups: self
                .workgroups
                .expect("Must set `workgroups` before building LaunchConfig"),
            threads_per_workgroup: self
                .threads_per_workgroup
                .expect("Must set `threads_per_workgroup` before building LaunchConfig"),
        }
    }
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
impl Module {
    #[doc(hidden)]
    pub fn new(data: &[u8]) -> Self {
        #[cfg(feature = "amd")]
        unsafe {
            let mut module: hip_runtime_sys::hipModule_t = std::ptr::null_mut();
            let result = hip_runtime_sys::hipModuleLoadData(
                &mut module,
                data.as_ptr() as *const std::ffi::c_void,
            );
            assert_eq!(result, hip_runtime_sys::hipError_t::hipSuccess);
            Self { module }
        }
    }

    #[doc(hidden)]
    pub fn get_kernel(&self, name: &str) -> Kernel {
        #[cfg(feature = "amd")]
        {
            use std::ffi;

            unsafe {
                let mut function: hip_runtime_sys::hipFunction_t = std::ptr::null_mut();
                let kernel_name = std::ffi::CString::new(name).expect("Invalid kernel name");
                let result = hip_runtime_sys::hipModuleGetFunction(
                    &mut function,
                    self.module,
                    kernel_name.as_ptr(),
                );
                assert_eq!(
                    result,
                    hip_runtime_sys::hipError_t::hipSuccess,
                    "Failed to find kernel {:?}",
                    kernel_name
                );
                Kernel { func: function }
            }
        }
    }
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
impl Kernel {
    #[doc(hidden)]
    pub fn launch<T: 'static>(&self, launch_config: LaunchConfig, mut args: T) {
        #[cfg(feature = "amd")]
        {
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
                    launch_config.threads_per_workgroup[0],
                    launch_config.threads_per_workgroup[1],
                    launch_config.threads_per_workgroup[2],
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
}
