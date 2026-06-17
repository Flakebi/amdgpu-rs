#![cfg_attr(any(target_arch = "amdgpu", target_arch = "nvptx64"), no_std)]
// Allocators will potentially be stabilized before all the GPU necessary stuff.
#![cfg_attr(
    not(any(target_arch = "amdgpu", target_arch = "nvptx64")),
    feature(allocator_api)
)]

#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
use std::alloc::AllocError;
#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
use std::ptr::NonNull;

#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
use hip_runtime_sys::hipError_t::hipSuccess;

mod safe_kernel_arg;
pub use safe_kernel_arg::*;

pub use gpu_kernel_proc_macros::{kernel, kernel_lib_impl_dbg, kernel_lib_impl_rel};

#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
#[doc(hidden)]
pub use hip_runtime_sys;

#[cfg(any(target_arch = "amdgpu", target_arch = "nvptx64"))]
pub mod prelude {
    #[cfg(target_arch = "amdgpu")]
    pub use amdgpu_device_libs::prelude::{print, println};
}

// TODO cfg(any(doc))
#[cfg(any(doc, target_arch = "amdgpu"))]
pub mod intrinsics {
    pub use amdgpu_device_libs::dispatch_ptr;
    pub use amdgpu_device_libs::prelude::{
        s_barrier, workgroup_id_x, workgroup_id_y, workgroup_id_z, workitem_id_x, workitem_id_y,
        workitem_id_z,
    };
}

#[macro_export]
macro_rules! kernel_lib {
    () => {
        // TODO Different calls for debug and release mode
        #[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
        #[cfg(debug_assertions)]
        ::gpu_kernel::kernel_lib_impl_dbg!();
        #[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
        #[cfg(not(debug_assertions))]
        ::gpu_kernel::kernel_lib_impl_rel!();

        #[cfg(any(target_arch = "amdgpu", target_arch = "nvptx64"))]
        extern crate alloc;
    };
}

#[non_exhaustive]
#[derive(Clone, Default, Eq, Hash, PartialEq)]
pub struct LaunchConfig {
    pub workgroups: Option<[u32; 3]>,
    pub threads_per_workgroup: Option<[u32; 3]>,
}

/// Allocate managed memory on AMD that lives on the CPU and is visible to the GPU as well.
///
/// On GPUs that support it (mostly MI cards), managed memory can be automatically transferred
/// between CPU and GPU.
/// See the [unified memory management] documentation.
///
/// [unified memory management]: https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html
#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
pub struct ManagedMemAlloc;

/// Define global allocator.
#[cfg(all(
    feature = "amd-allocator",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
#[global_allocator]
static HEAP: ManagedMemAlloc = ManagedMemAlloc;

/// Allocate memory on the GPU, visible to the CPU as well.
#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
pub struct GpuAlloc;

#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
pub type GpuBox<T, A = GpuAlloc> = Box<T, A>;

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
#[doc(hidden)]
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

#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
struct HipStream(hip_runtime_sys::hipStream_t);

#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
thread_local! {
    static STREAM: std::cell::RefCell<HipStream> = std::cell::RefCell::new(HipStream::new());
}

impl LaunchConfig {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn workgroups(&mut self, workgroups: [u32; 3]) -> &mut Self {
        self.workgroups = Some(workgroups);
        self
    }

    pub fn threads_per_workgroup(&mut self, threads_per_workgroup: [u32; 3]) -> &mut Self {
        self.threads_per_workgroup = Some(threads_per_workgroup);
        self
    }
}

#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
unsafe impl std::alloc::GlobalAlloc for ManagedMemAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        use std::ffi;
        unsafe {
            let mut ptr: *mut ffi::c_void = std::ptr::null_mut();
            let result = hip_runtime_sys::hipMallocManaged(
                &mut ptr,
                layout.size(),
                hip_runtime_sys::hipMemAttachGlobal,
            );
            assert_eq!(result, hipSuccess);
            ptr as *mut _
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _: std::alloc::Layout) {
        unsafe {
            let result = hip_runtime_sys::hipFree(ptr as *mut _);
            assert_eq!(result, hipSuccess);
        };
    }
}

#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
unsafe impl std::alloc::Allocator for GpuAlloc {
    #[inline]
    fn allocate(&self, layout: std::alloc::Layout) -> Result<NonNull<[u8]>, AllocError> {
        use std::ffi;
        unsafe {
            let mut ptr: *mut ffi::c_void = std::ptr::null_mut();
            let result = hip_runtime_sys::hipMalloc(&mut ptr, layout.size());
            assert_eq!(result, hipSuccess);
            Ok(NonNull::slice_from_raw_parts(
                NonNull::new(ptr as *mut _).ok_or(AllocError)?,
                layout.size(),
            ))
        }
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, _: std::alloc::Layout) {
        unsafe {
            let result = hip_runtime_sys::hipFree(ptr.as_ptr() as *mut _);
            assert_eq!(result, hipSuccess);
        };
    }
}

#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
impl HipStream {
    fn new() -> Self {
        unsafe {
            let mut stream: hip_runtime_sys::hipStream_t = std::ptr::null_mut();
            let result = hip_runtime_sys::hipStreamCreate(&mut stream);
            assert_eq!(result, hipSuccess);
            Self(stream)
        }
    }
}

#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
impl Drop for HipStream {
    fn drop(&mut self) {
        unsafe {
            let result = hip_runtime_sys::hipStreamDestroy(self.0);
            assert_eq!(result, hipSuccess);
        }
    }
}

#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
fn thread_local_stream() -> hip_runtime_sys::hipStream_t {
    STREAM.with_borrow(|s| s.0)
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
impl Module {
    pub fn new(data: &[u8]) -> Self {
        #[cfg(feature = "amd")]
        unsafe {
            let mut module: hip_runtime_sys::hipModule_t = std::ptr::null_mut();
            let result = hip_runtime_sys::hipModuleLoadData(
                &mut module,
                data.as_ptr() as *const std::ffi::c_void,
            );
            assert_eq!(result, hipSuccess);
            Self { module }
        }
    }

    pub fn get_kernel(&self, name: &str) -> Kernel {
        #[cfg(feature = "amd")]
        unsafe {
            let mut function: hip_runtime_sys::hipFunction_t = std::ptr::null_mut();
            let kernel_name = std::ffi::CString::new(name).expect("Invalid kernel name");
            let result = hip_runtime_sys::hipModuleGetFunction(
                &mut function,
                self.module,
                kernel_name.as_ptr(),
            );
            assert_eq!(
                result, hipSuccess,
                "Failed to find kernel {:?}",
                kernel_name
            );
            Kernel { func: function }
        }
    }
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
impl Kernel {
    #[cfg(feature = "amd")]
    pub fn func(&self) -> hip_runtime_sys::hipFunction_t {
        self.func
    }

    pub unsafe fn launch_impl<T: ?Sized>(&self, launch_config: &LaunchConfig, args: &mut T) {
        #[cfg(feature = "amd")]
        {
            use std::ffi;

            let mut size = std::mem::size_of_val(args);

            #[allow(clippy::manual_dangling_ptr)]
            let mut config = [
                0x1 as *mut ffi::c_void,                          // Next come arguments
                args as *mut _ as *mut ffi::c_void,               // Pointer to arguments
                0x2 as *mut ffi::c_void,                          // Next comes size
                std::ptr::addr_of_mut!(size) as *mut ffi::c_void, // Pointer to size of arguments
                0x3 as *mut ffi::c_void,                          // End
            ];

            let workgroups = launch_config
                .workgroups
                .expect("Must set `workgroups` in LaunchConfig");
            let threads_per_workgroup = launch_config
                .threads_per_workgroup
                .expect("Must set `threads_per_workgroup` in LaunchConfig");

            unsafe {
                let stream = thread_local_stream();
                // Launch two workgroups (2x1x1), each of the size (LEN/2)x1x1
                let result = hip_runtime_sys::hipModuleLaunchKernel(
                    self.func,
                    workgroups[0],
                    workgroups[1],
                    workgroups[2],
                    threads_per_workgroup[0],
                    threads_per_workgroup[1],
                    threads_per_workgroup[2],
                    0,                    // sharedMemBytes for extern shared variables
                    stream,               // stream
                    std::ptr::null_mut(), // params (unimplemented in hip)
                    config.as_mut_ptr(),  // arguments
                );
                assert_eq!(result, hipSuccess, "Failed to launch kernel");

                let result = hip_runtime_sys::hipStreamSynchronize(stream);
                assert_eq!(result, hipSuccess, "Failed to wait for kernel to finish");
            }
        }
    }
}
