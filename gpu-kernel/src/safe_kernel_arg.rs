use core::marker::PhantomData;

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
use crate::{GpuBox, LaunchConfig};

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
macro_rules! safe_kernel_arg_impl {
    ($($ty:ty),*) => {
        $(
            #[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
            unsafe impl SafeKernelArg for $ty {
                type Output = Self;

                fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
                    self
                }
            }
        )*
    };
}

/// Marker trait for types that are safe to pass to GPU kernels.
///
/// The `Output` type that is passed to the GPU can be the same as the type
/// the trait is implemented for or it can be different.
/// This is useful to e.g. allow an allocated `Vec<T>` to be passed as a `&[T]`
/// but not allow passing a slice directly as it might not point to memory that
/// is readable by the GPU.
///
/// # Safety
///
/// An implementor guarantees that a GPU kernel receiving the output can freely
/// use it in safe code.
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
#[diagnostic::on_unimplemented(
    message = "`SafeKernelArg` is not implemented for `{Self}`",
    label = "`{Self}` is passed to a kernel here",
    note = "All kernel arguments must implement `SafeKernelArg` or the kernel must be marked as `unsafe`"
)]
pub unsafe trait SafeKernelArg {
    /// The type that is passed to the GPU.
    type Output;

    /// Convert into the actual GPU argument.
    ///
    /// May panic if necessary constraints are violated.
    fn into_kernel_arg(self, launch_config: &LaunchConfig) -> Self::Output;
}

/// Pass an `&mut Vec<T>` to a kernel and each thread gets mutable access to one element of the vector.
///
/// The size of the vector needs to be equal to the number of launched threads otherwise launching the
/// kernel panics.
// Needs repr transparent to be passed as a pointer. Structs would be passed by reference.
#[repr(transparent)]
pub struct ThreadIndexedSlice<'a, T> {
    ptr: *mut T,
    phantom: PhantomData<&'a mut T>,
}

// SAFETY: These primitive types have the same layout in the CPU and GPU calling convention.
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
safe_kernel_arg_impl!(bool, u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

// SAFETY: A pointer has the same layout in the CPU and GPU calling convention.
// It might not point to GPU accessible memory, but that is fine as it is not
// safely dereferenceable.
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl<T> SafeKernelArg for *const T {
    type Output = Self;

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self
    }
}

// SAFETY: See *const T
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl<T> SafeKernelArg for *mut T {
    type Output = Self;

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self
    }
}

// SAFETY: When using the allocator, heap memory is visible to the GPU, so the
// slice is readable if `T` has the same layout on the GPU.
#[cfg(all(
    feature = "amd-allocator",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
unsafe impl<'a, T: SafeKernelArg<Output = T>> SafeKernelArg for &'a Vec<T> {
    type Output = &'a [T];

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self.as_slice()
    }
}

// SAFETY: See Vec<T>
#[cfg(all(
    feature = "amd-allocator",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
unsafe impl<'a> SafeKernelArg for &'a String {
    type Output = &'a str;

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self.as_str()
    }
}

// SAFETY: See Vec<T>
#[cfg(all(
    feature = "amd-allocator",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
unsafe impl<'a, T: SafeKernelArg<Output = T>> SafeKernelArg for &'a Box<T> {
    type Output = &'a T;

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self.as_ref()
    }
}

// SAFETY: See Vec<T>
#[cfg(all(
    feature = "amd-allocator",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
unsafe impl<'a, T: SafeKernelArg<Output = T>> SafeKernelArg for &'a Box<[T]> {
    type Output = &'a [T];

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self.as_ref()
    }
}

// SAFETY: See Vec<T>
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl<'a, T: SafeKernelArg<Output = T>> SafeKernelArg for &'a GpuBox<T> {
    type Output = &'a T;

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self.as_ref()
    }
}

// SAFETY: See Vec<T>
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl<'a, T: SafeKernelArg<Output = T>> SafeKernelArg for &'a GpuBox<[T]> {
    type Output = &'a [T];

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self.as_ref()
    }
}

// SAFETY: See Vec<T>
#[cfg(all(
    feature = "amd-allocator",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
unsafe impl<'a, T: SafeKernelArg<Output = T>> SafeKernelArg for &'a std::sync::Arc<T> {
    type Output = &'a T;

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        std::ops::Deref::deref(self)
    }
}

/// Implement SafeKernelArg<Output = ThreadIndexedSlice<T>> for a list type
macro_rules! safe_kernel_arg_list_impl {
    ($ty:ty: $len:expr; $ptr:expr) => {
        #[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
        unsafe impl<'a, T: SafeKernelArg<Output = T>> SafeKernelArg for &'a mut $ty {
            type Output = ThreadIndexedSlice<'a, T>;

            fn into_kernel_arg(self, launch_config: &LaunchConfig) -> ThreadIndexedSlice<'a, T> {
                // assert that vector is long enough for launched threads
                let launch_size = launch_config
                    .threads_per_workgroup
                    .unwrap()
                    .iter()
                    .map(|i| *i as usize)
                    .product::<usize>()
                    * launch_config
                        .workgroups
                        .unwrap()
                        .iter()
                        .map(|i| *i as usize)
                        .product::<usize>();
                assert!(
                    $len(self) >= launch_size as usize,
                    "Passed vector is not large enough for the number of launched threads. Expected at least {launch_size} but got {}",
                    $len(self)
                );
                ThreadIndexedSlice {
                    ptr: $ptr(self),
                    phantom: PhantomData,
                }
            }
        }
    };
}

// SAFETY: See Vec<T>
#[cfg(feature = "amd-allocator")]
safe_kernel_arg_list_impl!(Vec<T>: |v: &[_]| v.len(); |v: &mut [_]| v.as_mut_ptr());
#[cfg(feature = "amd-allocator")]
safe_kernel_arg_list_impl!(Box<[T]>: |v: &[_]| v.len(); |v: &mut [_]| v.as_mut_ptr());
safe_kernel_arg_list_impl!(GpuBox<[T]>: |v: &[_]| v.len(); |v: &mut [_]| v.as_mut_ptr());

#[cfg(any(target_arch = "amdgpu", target_arch = "nvptx64"))]
fn thread_id() -> usize {
    use crate::intrinsics::*;
    let dispatch = crate::intrinsics::dispatch_ptr();

    // Compute size as ((z * dimY) + y) * dimX + x
    let mut id =
        workitem_id_z() as usize + dispatch.workgroup_size_z as usize * workgroup_id_z() as usize;
    id *= dispatch.grid_size_y as usize;
    id += workitem_id_y() as usize + dispatch.workgroup_size_y as usize * workgroup_id_y() as usize;
    id *= dispatch.grid_size_x as usize;
    id += workitem_id_x() as usize + dispatch.workgroup_size_x as usize * workgroup_id_x() as usize;
    id
}

impl<'a, T> ThreadIndexedSlice<'a, T> {
    /// Constructs a `ThreadIndexedSlice` from a raw base pointer.
    ///
    /// # Safety
    ///
    /// - `ptr` must point to at least number of GPU threads consecutive properly initialized values of type T.
    /// - No constant or mutable reference to the data must exist for the lifetime of this struct.
    pub unsafe fn from_ptr(ptr: *mut T) -> Self {
        Self {
            ptr,
            phantom: PhantomData,
        }
    }

    /// Returns the base pointer of the wrapped list.
    pub fn as_mut_base_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Get a reference to the element for the current thread index.
    #[cfg(any(doc, target_arch = "amdgpu", target_arch = "nvptx64"))]
    pub fn get(&self) -> &T {
        unsafe { &*self.ptr.add(thread_id()) }
    }

    /// Get a mutable reference to the element for the current thread index.
    #[cfg(any(doc, target_arch = "amdgpu", target_arch = "nvptx64"))]
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.ptr.add(thread_id()) }
    }
}
