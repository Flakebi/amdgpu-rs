#![cfg_attr(any(target_arch = "amdgpu", target_arch = "nvptx64"), no_std)]

#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
use hip_runtime_sys::hipError_t::hipSuccess;

pub use gpu_kernel_proc_macros::{kernel, kernel_lib_impl};

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
#[cfg(target_arch = "amdgpu")]
pub use amdgpu_device_libs::{dispatch_ptr, intrinsics};

#[macro_export]
macro_rules! kernel_lib {
    () => {
        #[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
        ::gpu_kernel::kernel_lib_impl!();

        #[cfg(any(target_arch = "amdgpu", target_arch = "nvptx64"))]
        extern crate alloc;
    };
}

// hipCpuDeviceId
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
const HIP_CPU_DEVICE_ID: std::ffi::c_int = -1;

#[non_exhaustive]
#[derive(Clone, Default, Eq, Hash, PartialEq)]
pub struct LaunchConfig {
    pub workgroups: Option<[u32; 3]>,
    pub threads_per_workgroup: Option<[u32; 3]>,
}

#[cfg(all(
    feature = "amd-allocator",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
pub struct AmdAllocator;

#[cfg(all(
    feature = "amd-allocator",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
unsafe impl std::alloc::GlobalAlloc for AmdAllocator {
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

/// Define global allocator.
#[cfg(all(
    feature = "amd-allocator",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
#[global_allocator]
static HEAP: AmdAllocator = AmdAllocator;

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
enum PrefetchType {
    /// No prefetching.
    Raw,
    /// Prefetch to the GPU.
    Prefetch,
    /// Prefetch to the GPU and mark as incoherent/coarse grain.
    PrefetchIncoherent,
}

/// Prefetch memory to the GPU.
///
/// Defaults to not preloading, see [`PrefetchMem::raw`].
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
#[derive(Clone, Copy)]
pub struct PrefetchMem<'a, T: ?Sized> {
    inner: &'a T,
    ty: PrefetchType,
}

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
impl<'a, T: ?Sized> PrefetchMem<'a, T> {
    // TODO Add hipHostRegister + hipHostGetDevicePointer to share (aligned) constant memory

    /// Pass through reference unmodified.
    ///
    /// This is the default.
    pub fn raw(inner: &'a T) -> Self {
        Self {
            inner,
            ty: PrefetchType::Raw,
        }
    }

    /// Use `prefetch_incoherent` for more performance if the CPU does not need coherent access to the memory while a kernel is running.
    pub fn prefetch(inner: &'a T) -> Self {
        Self {
            inner,
            ty: PrefetchType::Prefetch,
        }
    }

    pub fn prefetch_incoherent(inner: &'a T) -> Self {
        Self {
            inner,
            ty: PrefetchType::PrefetchIncoherent,
        }
    }

    /// Must run on the same thread as the kernel using the memory is launched to be properly synchronized.
    pub unsafe fn apply_async(&self) {
        if self.ty == PrefetchType::Raw {
            return;
        }
        #[cfg(feature = "amd")]
        {
            unsafe {
                let stream = thread_local_stream();
                let mut device = 0;
                let result = hip_runtime_sys::hipGetDevice(&mut device as *mut _);
                assert_eq!(result, hipSuccess);

                let result = hip_runtime_sys::hipMemAdvise(
                    self.inner as *const _ as *const _,
                    std::mem::size_of_val(self.inner),
                    hip_runtime_sys::hipMemoryAdvise::hipMemAdviseSetAccessedBy,
                    device,
                );
                assert_eq!(result, hipSuccess);

                let result = hip_runtime_sys::hipMemAdvise(
                    self.inner as *const _ as *const _,
                    std::mem::size_of_val(self.inner),
                    hip_runtime_sys::hipMemoryAdvise::hipMemAdviseSetReadMostly,
                    device,
                );
                assert_eq!(result, hipSuccess);

                let result = hip_runtime_sys::hipMemAdvise(
                    self.inner as *const _ as *const _,
                    std::mem::size_of_val(self.inner),
                    hip_runtime_sys::hipMemoryAdvise::hipMemAdviseSetPreferredLocation,
                    device,
                );
                assert_eq!(result, hipSuccess);

                if self.ty == PrefetchType::PrefetchIncoherent {
                    let result = hip_runtime_sys::hipMemAdvise(
                        self.inner as *const _ as *const _,
                        std::mem::size_of_val(self.inner),
                        hip_runtime_sys::hipMemoryAdvise::hipMemAdviseSetCoarseGrain,
                        device,
                    );
                    assert_eq!(result, hipSuccess);
                }

                println!(
                    "Prefetching {} bytes to device {device}",
                    std::mem::size_of_val(self.inner)
                );
                let result = hip_runtime_sys::hipMemPrefetchAsync(
                    self.inner as *const _ as *const _,
                    std::mem::size_of_val(self.inner),
                    device,
                    stream,
                );
                assert_eq!(result, hipSuccess);
            }
        }
    }

    pub fn apply(&self) {
        #[cfg(feature = "amd")]
        {
            unsafe {
                self.apply_async();

                let stream = thread_local_stream();
                let result = hip_runtime_sys::hipStreamSynchronize(stream);
                assert_eq!(result, hipSuccess, "Failed to wait for kernel to finish");
            }
        }
    }

    /// Prefetch back to the CPU if it was prefetched into GPU memory.
    pub fn unapply(&self) {
        if self.ty == PrefetchType::Raw {
            return;
        }
        return;
        #[cfg(feature = "amd")]
        {
            unsafe {
                let stream = thread_local_stream();
                let mut device = 0;
                let result = hip_runtime_sys::hipGetDevice(&mut device as *mut _);
                assert_eq!(result, hipSuccess);

                let result = hip_runtime_sys::hipMemPrefetchAsync(
                    self.inner as *const _ as *const _,
                    std::mem::size_of_val(self.inner),
                    HIP_CPU_DEVICE_ID,
                    stream,
                );
                assert_eq!(result, hipSuccess);

                let result = hip_runtime_sys::hipMemAdvise(
                    self.inner as *const _ as *const _,
                    std::mem::size_of_val(self.inner),
                    hip_runtime_sys::hipMemoryAdvise::hipMemAdviseUnsetAccessedBy,
                    device,
                );
                assert_eq!(result, hipSuccess);

                if self.ty == PrefetchType::PrefetchIncoherent {
                    let result = hip_runtime_sys::hipMemAdvise(
                        self.inner as *const _ as *const _,
                        std::mem::size_of_val(self.inner),
                        hip_runtime_sys::hipMemoryAdvise::hipMemAdviseUnsetCoarseGrain,
                        device,
                    );
                    assert_eq!(result, hipSuccess);
                }

                let result = hip_runtime_sys::hipStreamSynchronize(stream);
                assert_eq!(result, hipSuccess, "Failed to wait for kernel to finish");
            }
        }
    }
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
impl<'a, T> PrefetchMem<'a, [T]> {
    pub unsafe fn copy(inner: &'a [T]) -> Self {
        #[cfg(feature = "amd")]
        {
            unsafe {
                use core::slice;

                let size = std::mem::size_of_val(inner);
                let mut mem = std::ptr::null_mut();
                let result = hip_runtime_sys::hipMalloc(&mut mem, size);
                assert_eq!(result, hipSuccess);

                let result = hip_runtime_sys::hipMemcpy(
                    mem,
                    inner as *const _ as *const _,
                    size,
                    hip_runtime_sys::hipMemcpyKind::hipMemcpyHostToDevice,
                );
                assert_eq!(result, hipSuccess);

                Self {
                    inner: slice::from_raw_parts(mem as *const _, inner.len()),
                    ty: PrefetchType::Raw,
                }
            }
        }
    }
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
impl<'a, T: ?Sized> std::ops::Deref for PrefetchMem<'a, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
impl<'a, T: ?Sized> From<&'a T> for PrefetchMem<'a, T> {
    fn from(inner: &'a T) -> Self {
        Self {
            inner,
            ty: PrefetchType::Raw,
        }
    }
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
