//! Running Rust code on a GPU is not as hard as it might sound and here is how it’s done!
//!
//! Let us start with the code, it takes just a few lines:
//! ```rust,no_run
//! // main.rs
//! // GPU code is no-std and requires the nightly gpu_kernel ABI
//! #![cfg_attr(feature = "gpu", no_std, feature(abi_gpu_kernel))]
//!
//! // Macro to compile and include the GPU code
//! gpu_kernel::kernel_lib!();
//!
//! // Define a kernel, this function runs on the GPU
//! #[gpu_kernel::kernel]
//! fn kernel(s: &str) {
//!     let id = gpu_kernel::intrinsics::workitem_id_x();
//!     println!("Hello {s} from thread #{}!", id);
//! }
//!
//! #[cfg(not(feature = "gpu"))]
//! fn main() {
//!     let s = "World".to_string();
//!     // Launch 10 threads on the GPU
//!     kernel.launch(
//!         gpu_kernel::LaunchConfig::new()
//!             .threads_per_workgroup([10, 1, 1])
//!             .workgroups([1, 1, 1]),
//!         &s,
//!     );
//! }
//! ```
//!
//! This is all Rust code, it prints hello world from the GPU for each started thread:
//! ```bash
//! $ cargo run
//! Hello World from thread #0!
//! Hello World from thread #1!
//! Hello World from thread #2!
//! Hello World from thread #3!
//! Hello World from thread #4!
//! Hello World from thread #5!
//! Hello World from thread #6!
//! Hello World from thread #7!
//! Hello World from thread #8!
//! Hello World from thread #9!
//! ```
//!
//! In `Cargo.toml`, we add `gpu-kernel` as a dependency and that’s it:
//! ```toml
//! # Cargo.toml
//! [package]
//! name = "hello_world"
//! version = "0.1.0"
//! edition = "2024"
//!
//! # This gets defined when building for the gpu.
//! # It can be omitted when using target_arch or similar for cfg conditions, it exists for convenience only.
//! [features]
//! gpu = []
//!
//! [dependencies]
//! gpu-kernel = "0.1"
//! ```
//!
//! For `cargo run` to work, the GPU compute runtime needs to be installed, see the next section.
//!
//! ## Setup
//!
//! Currently, AMD GPUs are supported.
//! Contributions for other Rust GPU targets are welcome, adding support to `gpu-kernel` should be relatively straightforward.
//!
//! Nightly Rust is currently required for the gpu_kernel ABI and GPU intrinsics.
//!
//! 1. Install ROCm. On Ubuntu 26.04, this is a simple `apt install rocm-dev`
//! 1. Add `rust-src` to rustup to support build-std: `rustup component add rust-src`
//! 1. Configure your GPU in cargo’s config, find your version with `rocminfo | grep gfx`:
//!    ```toml
//!    # ~/.cargo/config.toml
//!    [target.amdgcn-amd-amdhsa]
//!    rustflags = ["-Ctarget-cpu=gfx<your version>"]
//!    # If rocminfo shows xnack- for your GPU, add "-Ctarget-feature=-xnack-support" as well
//!    ```
//!    Alternatively, specify the flags through an environment variable: `CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS=-Ctarget-cpu=gfx<your version>`
//! 1. Set `HIP_PATH=/usr` for `hip-runtime-sys` to find the hip headers
//!
//! On NixOS, skip step 4 and add `rocmPackages.clr` to your dev shell to automagically set `HIP_DEVICE_LIB_PATH` and `HIP_PATH` or manually set `HIP_DEVICE_LIB_PATH="${rocmPackages.rocm-device-libs}/amdgcn/bitcode"` and `HIP_PATH="${rocmPackages.clr}"`.
//!
//! ## Settings
//!
//! Configuration files like `.cargo/config.toml` and `~/.cargo/config.toml` can be used to specify compiler flags as described in the [setup](#setup) section.
//!
//! Additionally, a few of environment variables can be set:
//!
//! | Env variable                               | Default                                         | Example               | Description                                                              |
//! |--------------------------------------------|-------------------------------------------------|-----------------------|--------------------------------------------------------------------------|
//! | `HIP_PATH`                                 | `/opt/rocm/hip`                                 | `/usr`                | Path to the hip installation to find headers                             |
//! | `HIP_DEVICE_LIB_PATH`                      | `$(hipconfig -l)/../lib/clang/*/amdgcn/bitcode` |                       | Path to device libs, ends with `amdgcn/bitcode` and contains `.bc` files |
//! | `CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS` | empty                                           | `-Ctarget-cpu=gfx900` | RUSTFLAGS used to compile amdgpu GPU code                                |
//! | `CARGO_TARGET_AMDGCN_AMD_AMDHSA_FLAGS`     | empty                                           | `-v`                  | Cargo flags used to compile amdgpu GPU code                              |
//!
//! Several flags are added automatically to the GPU compilation.
//!
//! - If a `gpu` feature is defined in `Cargo.toml`, `--features=gpu` is passed to cargo
//! - The `crate-type` is set to `cdylib`
//! - Device libs are added to `link-arg`s and `-Clinker-plugin-lto` is enabled
//! - core and alloc are built with `-Zbuild-std=core,alloc`
//! - In debug mode, `opt-level=2` is set, as no optimizations can lead to crashes or compilation failures in the backend
//! - In release mode, `panic=immediate-abort` is set for performance, so no panic messages are available
#![deny(missing_docs)]
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

pub use gpu_kernel_proc_macros::kernel;
#[doc(hidden)]
pub use gpu_kernel_proc_macros::{kernel_lib_impl_dbg, kernel_lib_impl_rel};

#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
#[doc(hidden)]
pub use hip_runtime_sys;

/// Items automatically imported for kernels.
///
/// These don’t appear in the docs as they are only available in GPU code.
#[cfg(any(doc, target_arch = "amdgpu", target_arch = "nvptx64"))]
pub mod prelude {
    #[cfg(target_arch = "amdgpu")]
    pub use amdgpu_device_libs::prelude::{print, println};
}

/// Some basic, useful intrinsics for GPU kernels.
///
/// Once there is more support in `core`, this will be removed.
///
/// These don’t appear in the docs as they are only available in GPU code.
#[cfg(any(doc, target_arch = "amdgpu"))]
pub mod intrinsics {
    #[cfg(target_arch = "amdgpu")]
    pub use amdgpu_device_libs::dispatch_ptr;
    #[cfg(target_arch = "amdgpu")]
    pub use amdgpu_device_libs::prelude::{
        s_barrier, workgroup_id_x, workgroup_id_y, workgroup_id_z, workitem_id_x, workitem_id_y,
        workitem_id_z,
    };
}

/// The `kernel_lib!()` macro declares a crate as a library of GPU kernels.
///
/// It compiles the crate for the GPU and includes the compiled binary on the CPU side.
///
/// # Example
///
/// ```
/// // Somewhere at the top-level of your crate
/// gpu_kernel::kernel_lib!();
/// ```
#[macro_export]
macro_rules! kernel_lib {
    () => {
        // Different calls for debug and release mode, so the kernels are compiled appropriately
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

/// A GPU kernel never comes alone, it is always groups of them that are launched together.
///
/// A number of launched threads together are called a workgroup.
/// And a number of workgroups are launched together.
/// This struct specifies how many workgroups are launched and how many threads are contained in each of them.
///
/// The total number of threads launched is number of workgroups times threads per workgroup.
///
/// # Example
///
/// ```
/// # use gpu_kernel::LaunchConfig;
/// let launch_config = LaunchConfig::new()
///     .workgroups([1, 0, 0])
///     .threads_per_workgroup([1, 0, 0]);
/// ```
#[non_exhaustive]
#[derive(Clone, Default, Eq, Hash, PartialEq)]
pub struct LaunchConfig {
    /// The number of workgroups launched on the GPU.
    ///
    /// A three-dimensional size for x, y, z dimensions.
    /// For a simple list of threads, this can be `[n, 1, 1]`.
    pub workgroups: Option<[u32; 3]>,
    /// The number of threads in each workgroup.
    ///
    /// A three-dimensional size for x, y, z dimensions.
    /// For a simple list of threads, this can be `[n, 1, 1]`.
    pub threads_per_workgroup: Option<[u32; 3]>,
}

/// Allocate managed memory on AMD that lives on the CPU and is visible to the GPU as well.
///
/// On GPUs that support it (mostly MI cards), managed memory can be automatically transferred
/// between CPU and GPU.
/// See the [unified memory management] documentation.
///
/// With the `amd-allocator` crate feature (enabled by default), this is the default allocator.
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
///
/// [`GpuBox`] is a convenient `Box` using this allocator.
#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
pub struct GpuAlloc;

/// A `Box` allocated on the GPU, also accessible from the CPU.
///
/// # Example
///
/// ```
/// # use gpu_kernel::GpuBox;
/// // This integer is allocated in GPU memory,
/// // so fast to access on the GPU and slow to access on the CPU.
/// let gpu_int = GpuBox::new(42);
/// ```
#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
pub type GpuBox<T, A = GpuAlloc> = Box<T, A>;

/// A loaded, compiled GPU binary.
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

/// A loaded, compiled GPU kernel.
///
/// Can be launched on the GPU.
///
/// The `#[kernel]` macro adds a `launch` function that takes a [`&LaunchConfig`](`LaunchConfig`)
/// as first argument and all kernel arguments afterwards.
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
    /// A thread-local stream to launch and wait for kernels.
    static STREAM: std::cell::RefCell<HipStream> = std::cell::RefCell::new(HipStream::new());
}

impl LaunchConfig {
    /// Create an empty `LaunchConfig`.
    ///
    /// At least [`Self::workgroups`] and [`Self::threads_per_workgroup`] need to be filled out, otherwise launching panics.
    pub fn new() -> Self {
        Default::default()
    }

    /// The number of workgroups launched on the GPU.
    ///
    /// A three-dimensional size for x, y, z dimensions.
    /// For a simple list of threads, this can be `[n, 1, 1]`.
    pub fn workgroups(&mut self, workgroups: [u32; 3]) -> &mut Self {
        self.workgroups = Some(workgroups);
        self
    }

    /// The number of threads in each workgroup.
    ///
    /// A three-dimensional size for x, y, z dimensions.
    /// For a simple list of threads, this can be `[n, 1, 1]`.
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

/// Get the thread-local stream.
///
/// Internally copies the reference to make access simpler.
#[cfg(all(
    feature = "amd",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
fn thread_local_stream() -> hip_runtime_sys::hipStream_t {
    STREAM.with_borrow(|s| s.0)
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
impl Module {
    /// Load a module from a binary.
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

    /// Get the kernel with the specified name from the loaded binary.
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
    /// Get the raw kernel function.
    #[cfg(feature = "amd")]
    pub fn func(&self) -> hip_runtime_sys::hipFunction_t {
        self.func
    }

    /// Launch a kernel, passing the given type as arguments.
    ///
    /// # Safety
    ///
    /// `T` must be the actual arguments expected by the kernel.
    #[doc(hidden)]
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
