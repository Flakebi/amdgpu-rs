// Update:
// cargo readme > README.md
// cargo readme --template ../README.tpl > ../README.md

//! Support library for the amdgpu target.
//!
//! By default, the amdgpu target supports `core`, but not `std`.
//! `alloc` is supported when a global allocator is specified.
//!
//! `amdgpu-device-libs` brings some std-like features to the amdgpu target:
//!
//! - `print!()` and `println!()` macros for printing on the host stdout
//! - A global allocator to support `alloc`
//! - A panic handler
//! - Access to more intrinsics and device-libs functions
//!
//! All these features are enabled by default, but can be turned on selectively with `default-features = false, features = […]`.
//!
//! `amdgpu-device-libs` works by linking to the [ROCm device-libs](https://github.com/ROCm/llvm-project/tree/amd-staging/amd/device-libs) and a pre-compiled helper library.
//! The libraries are linked from a ROCm installation.
//! To make sure the libraries are found, set the environment variable `ROCM_PATH` or `ROCM_DEVICE_LIB_PATH` (higher priority if it is set).
//! It looks for `amdgcn/bitcode/*.bc` files in this path.
//!
//! ## Usage
//!
//! Create a new cargo library project and change it to compile a cdylib:
//! ```toml
//! # Cargo.toml
//! # Force lto
//! [profile.dev]
//! lto = true
//! [profile.release]
//! lto = true
//!
//! [lib]
//! # Compile a cdylib
//! crate-type = ["cdylib"]
//!
//! [build-dependencies]
//! # Used in build script to specify linker flags and link in device-libs
//! amdgpu-device-libs-build = { path = "../../amdgpu-device-libs-build" }
//!
//! [dependencies]
//! amdgpu-device-libs = { path = "../../amdgpu-device-libs" }
//! ```
//!
//! Add extra flags in `.cargo/config.toml`:
//! ```toml
//! # .cargo/config.toml
//! [build]
//! target = "amdgcn-amd-amdhsa"
//! # Enable linker-plugin-lto and workarounds
//! # Either add -Ctarget-cpu=gfx<version> here or specify it in CARGO_BUILD_RUSTFLAGS='-Ctarget-cpu=gfx<version>'
//! rustflags = ["-Clinker-plugin-lto", "-Zemit-thin-lto=no"]
//!
//! [unstable]
//! build-std = ["core", "alloc"]
//! ```
//!
//! And add a `build.rs` build script that links to the required libraries:
//! ```rust,ignore
//! // build.rs
//! fn main() {
//!     amdgpu_device_libs_build::build();
//! }
//! ```
//!
//! ## Example
//!
//! Minimal usage sample, see [`examples/println`](https://github.com/Flakebi/amdgpu-rs/tree/main/examples/println) for the full code.
//! ```rust
//! #![feature(abi_gpu_kernel)]
//! #![no_std]
//!
//! extern crate alloc;
//!
//! use alloc::vec::Vec;
//!
//! use amdgpu_device_libs::prelude::*;
//!
//! #[unsafe(no_mangle)]
//! pub extern "gpu-kernel" fn kernel(output: *mut u32) {
//!     let wg_id = workgroup_id_x();
//!     let id = workitem_id_x();
//!     let dispatch = dispatch_ptr();
//!     let complete_id = wg_id as usize * dispatch.workgroup_size_x as usize + id as usize;
//!     
//!     println!("Hello world from the GPU! (thread {wg_id}-{id})");
//!     
//!     let mut v = Vec::<u32>::new();
//!     for i in 0..100 {
//!         v.push(100 + i);
//!     }
//!     
//!     unsafe {
//!         *output.add(complete_id) = v[complete_id];
//!     }
//! }
//! ```
#![deny(missing_docs)]
#![allow(internal_features)]
#![feature(core_intrinsics, link_llvm_intrinsics)]
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use core::alloc::{GlobalAlloc, Layout};
use core::ffi;

/// Prints to the standard output.
///
/// Formats all arguments to [`format!`](alloc::format!).
#[cfg(feature = "print")]
#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {
        $crate::print(::alloc::format!($($arg)*));
    };
}

/// Prints to the standard output, with a newline.
///
/// Formats all arguments to [`format!`](alloc::format!) and appends a `\n`.
#[cfg(feature = "print")]
#[macro_export]
macro_rules! println {
    ($($arg:tt)*) => {
        let mut s = ::alloc::format!($($arg)*);
        s.push('\n');
        $crate::print(&s);
    };
}

pub mod intrinsics;

/// Prelude for functions that are generally useful when writing kernels.
///
/// Use as
/// ```rust
/// # #![no_std]
/// use amdgpu_device_libs::prelude::*;
/// ```
///
/// Contains `print!`, `println!`, intrinsics to get workitem and workgroup id among others.
pub mod prelude {
    #[cfg(feature = "device_libs")]
    pub use crate::dispatch_ptr;
    pub use crate::intrinsics::{
        s_barrier, workgroup_id_x, workgroup_id_y, workgroup_id_z, workitem_id_x, workitem_id_y,
        workitem_id_z,
    };
    #[cfg(feature = "print")]
    pub use print;
    #[cfg(feature = "print")]
    pub use println;
}

#[cfg(feature = "device_libs")]
unsafe extern "C" {
    #[cfg(feature = "hostcall")]
    #[allow(improper_ctypes)]
    fn __ockl_call_host_function(
        fptr: ffi::c_ulong,
        arg0: ffi::c_ulong,
        arg1: ffi::c_ulong,
        arg2: ffi::c_ulong,
        arg3: ffi::c_ulong,
        arg4: ffi::c_ulong,
        arg5: ffi::c_ulong,
        arg6: ffi::c_ulong,
    ) -> u128;

    // Functions implemented in HIP
    fn __amdgpu_util_alloc(size: ffi::c_ulong) -> *mut ffi::c_void;
    fn __amdgpu_util_dealloc(addr: *mut ffi::c_void);
    fn __amdgpu_util_print_stdout(s: *const ffi::c_char, size: ffi::c_int);

    // Intrinsics that return special addrspaces and therefore cannot be pure rust at the moment
    safe fn __amdgpu_util_dispatch_ptr() -> *const ffi::c_void;
    safe fn __amdgpu_util_queue_ptr() -> *mut ffi::c_void;
    safe fn __amdgpu_util_kernarg_segment_ptr() -> *const ffi::c_void;
    safe fn __amdgpu_util_implicitarg_ptr() -> *const ffi::c_void;
}

/// Handle to an HSA signal.
#[cfg(feature = "device_libs")]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[repr(C)]
pub struct HsaSignal {
    /// The internal representation of an HSA signal.
    pub handle: u64,
}

/// HSA packet to dispatch a kernel.
///
/// A pointer to the packet that was used to dispatch the currently running kernel can be obtained with [`dispatch_ptr`].
#[cfg(feature = "device_libs")]
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
#[repr(C)]
pub struct HsaKernelDispatchPacket {
    /// Packet header. Used to configure multiple packet parameters such as the
    /// packet type. The parameters are described by hsa_packet_header_t.
    pub header: u16,
    /// Dispatch setup parameters. Used to configure kernel dispatch parameters
    /// such as the number of dimensions in the grid. The parameters are described
    /// by hsa_kernel_dispatch_packet_setup_t.
    pub setup: u16,
    /// X dimension of work-group, in work-items. Must be greater than 0.
    pub workgroup_size_x: u16,
    /// Y dimension of work-group, in work-items. Must be greater than
    /// 0. If the grid has 1 dimension, the only valid value is 1.
    pub workgroup_size_y: u16,
    /// Z dimension of work-group, in work-items. Must be greater than
    /// 0. If the grid has 1 or 2 dimensions, the only valid value is 1.
    pub workgroup_size_z: u16,
    /// Reserved. Must be 0.
    pub reserved0: u16,
    /// X dimension of grid, in work-items. Must be greater than 0. Must
    /// not be smaller than @a workgroup_size_x.
    pub grid_size_x: u32,
    /// Y dimension of grid, in work-items. Must be greater than 0. If the grid has
    /// 1 dimension, the only valid value is 1. Must not be smaller than @a
    /// workgroup_size_y.
    pub grid_size_y: u32,
    /// Z dimension of grid, in work-items. Must be greater than 0. If the grid has
    /// 1 or 2 dimensions, the only valid value is 1. Must not be smaller than @a
    /// workgroup_size_z.
    pub grid_size_z: u32,
    /// Size in bytes of private memory allocation request (per work-item).
    pub private_segment_size: u32,
    /// Size in bytes of group memory allocation request (per work-group). Must not
    /// be less than the sum of the group memory used by the kernel (and the
    /// functions it calls directly or indirectly) and the dynamically allocated
    /// group segment variables.
    pub group_segment_size: u32,
    /// Opaque handle to a code object that includes an implementation-defined
    /// executable code for the kernel.
    pub kernel_object: u64,
    /// Pointer to the kernel arguments.
    pub kernarg_address: *mut ffi::c_void,
    /// Reserved. Must be 0.
    pub reserved2: u64,
    /// Signal used to indicate completion of the job. The application can use the
    /// special signal handle 0 to indicate that no signal is used.
    pub completion_signal: HsaSignal,
}

/// Panic handler.
///
/// Prints the panic message if the `print` feature is enabled.
/// Aborts the kernel.
#[cfg(feature = "panic_handler")]
#[cfg_attr(not(feature = "print"), allow(unused_variables))]
#[panic_handler]
#[inline]
fn panic(panic_info: &core::panic::PanicInfo) -> ! {
    #[cfg(feature = "print")]
    {
        use prelude::*;
        // workgroup x thread y panicked at …
        println!(
            "workgroup {},{},{} thread {},{},{} {panic_info}",
            workgroup_id_x(),
            workgroup_id_y(),
            workgroup_id_z(),
            workitem_id_x(),
            workitem_id_y(),
            workitem_id_z()
        );
    }

    core::intrinsics::abort();
}

/// The memory allocator of `device-libs`.
///
/// Allocates memory from the host through hostcalls to the HIP runtime in larger chunks and the subdivides them on the GPU.
#[cfg(feature = "alloc")]
pub struct Allocator;

#[cfg(feature = "alloc")]
unsafe impl GlobalAlloc for Allocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { __amdgpu_util_alloc(layout.size() as ffi::c_ulong) as *mut _ }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _: Layout) {
        unsafe { __amdgpu_util_dealloc(ptr as *mut _) };
    }
}

/// The bare print function underlying the `print!` macros.
///
/// Sends a string to the host console using the `printf` support of the HIP runtime.
///
/// # Example
///
/// ```rust
/// # #![no_std]
/// # fn main() {
/// amdgpu_device_libs::print("Printed on the host terminal\n");
/// # }
/// ```
#[cfg(feature = "print")]
#[inline]
pub fn print(s: &str) {
    unsafe {
        __amdgpu_util_print_stdout(
            s.as_ptr() as *const ffi::c_char,
            s.len().try_into().expect("String too long to print"),
        );
    }
}

/// Get the packet for this dispatch.
///
/// Get a reference to the packet that was used to dispatch this kernel.
/// The dispatch packet contains information like the workgroup size and dispatch size.
///
/// # Example
///
/// ```rust
/// # #![no_std]
/// # extern crate alloc;
/// # fn main() {
/// use amdgpu_device_libs::prelude::*;
///
/// let dispatch = dispatch_ptr();
/// println!("Workgroup size {}x{}x{}", dispatch.workgroup_size_x, dispatch.workgroup_size_y, dispatch.workgroup_size_z);
/// # }
/// ```
#[cfg(feature = "device_libs")]
#[inline]
pub fn dispatch_ptr() -> &'static HsaKernelDispatchPacket {
    unsafe { &*(__amdgpu_util_dispatch_ptr() as *const HsaKernelDispatchPacket) }
}

/// Call a function on the host.
///
/// This allows calling functions on the CPU from the GPU.
/// `function` must be the address of a function on the CPU.
/// Up to 7 64-bit arguments can be passed and two 64-bit values are returned.
///
/// The signature of the CPU function must be `fn(output: *mut u64, input: *const u64)`.
/// `output` points to two `u64` values for the return value and `input` points to seven `u64` for the function arguments.
///
/// The `function` pointer must be passed to the GPU through some mechanism like kernel arguments.
///
/// # Example
///
/// ```rust
/// # #![no_std]
/// # fn main() {
/// # let host_func = core::panic!();
/// // Get host_func from somewhere, e.g. arguments.
/// let arg0 = 42;
/// unsafe {
///     amdgpu_device_libs::call_host_function(host_func, arg0, 0, 0, 0, 0, 0, 0);
/// }
/// # }
/// ```
///
/// # Additional information
///
/// The CPU side is implemented here (`SERVICE_FUNCTION_CALL`): [ROCm/clr/rocclr/device/devhostcall.cpp](https://github.com/ROCm/clr/blob/f5b2516f5d8a44b06ad1907594db1be25a9fe57b/rocclr/device/devhostcall.cpp)  
/// The GPU side here: [ROCm/llvm-project/amd/device-libs/ockl/src/services.cl](https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/src/services.cl)
#[cfg(feature = "hostcall")]
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn call_host_function(
    function: u64,
    arg0: u64,
    arg1: u64,
    arg2: u64,
    arg3: u64,
    arg4: u64,
    arg5: u64,
    arg6: u64,
) -> u128 {
    unsafe { __ockl_call_host_function(function, arg0, arg1, arg2, arg3, arg4, arg5, arg6) }
}

// Define here, otherwise we may get undefined symbols.
// TODO implement in LLVM?
#[unsafe(no_mangle)]
#[inline]
extern "C" fn memcmp(s1: *const u8, s2: *const u8, n: usize) -> i32 {
    for i in 0..n {
        let diff = unsafe { i32::from(s1.add(i).read()) - i32::from(s2.add(i).read()) };
        if diff != 0 {
            return diff;
        }
    }
    0
}

/// Define global allocator.
#[cfg(feature = "global_allocator")]
#[global_allocator]
static HEAP: Allocator = Allocator;
