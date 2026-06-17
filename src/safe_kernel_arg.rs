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

/// Marker trait for types that are safe to pass to gpu kernels.
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
#[diagnostic::on_unimplemented(
    message = "`SafeKernelArg` is not implemented for `{Self}`",
    label = "`{Self}` is passed to a kernel here",
    note = "All kernel arguments must implement `SafeKernelArg` or the kernel must be marked as `unsafe`"
)]
pub unsafe trait SafeKernelArg {
    type Output;

    fn into_kernel_arg(self, launch_config: &LaunchConfig) -> Self::Output;
}

/// Pass an `&mut Vec<T>` to a kernel and each thread gets mutable access to one element of the vector.
///
/// The size of the vector needs to be equal to the number of launched threads otherwise launching the
/// kernel panics.
// Needs repr transparent to be passed as a pointer. Structs would be passed by reference.
#[repr(transparent)]
pub struct ThreadIndexedVec<'a, T> {
    ptr: *mut T,
    phantom: PhantomData<&'a mut T>,
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
safe_kernel_arg_impl!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl<T> SafeKernelArg for *const T {
    type Output = Self;

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self
    }
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl<T> SafeKernelArg for *mut T {
    type Output = Self;

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self
    }
}

#[cfg(all(
    feature = "amd-allocator",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
unsafe impl<'a, T: SafeKernelArg> SafeKernelArg for &'a Vec<T> {
    type Output = &'a [T];

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self.as_slice()
    }
}

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

#[cfg(all(
    feature = "amd-allocator",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
unsafe impl<'a, T: SafeKernelArg> SafeKernelArg for &'a Box<T> {
    type Output = &'a T;

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self.as_ref()
    }
}

#[cfg(all(
    feature = "amd-allocator",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
unsafe impl<'a, T: SafeKernelArg> SafeKernelArg for &'a Box<[T]> {
    type Output = &'a [T];

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self.as_ref()
    }
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl<'a, T: SafeKernelArg> SafeKernelArg for &'a GpuBox<T> {
    type Output = &'a T;

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self.as_ref()
    }
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl<'a, T: SafeKernelArg> SafeKernelArg for &'a GpuBox<[T]> {
    type Output = &'a [T];

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        self.as_ref()
    }
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
unsafe impl<'a, T: SafeKernelArg> SafeKernelArg for &'a std::sync::Arc<T> {
    type Output = &'a T;

    fn into_kernel_arg(self, _: &LaunchConfig) -> Self::Output {
        std::ops::Deref::deref(self)
    }
}

#[cfg(all(
    feature = "amd-allocator",
    not(any(target_arch = "amdgpu", target_arch = "nvptx64"))
))]
unsafe impl<'a, T: SafeKernelArg> SafeKernelArg for &'a mut Vec<T> {
    type Output = ThreadIndexedVec<'a, T>;

    fn into_kernel_arg(self, launch_config: &LaunchConfig) -> ThreadIndexedVec<'a, T> {
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
            self.len() >= launch_size as usize,
            "Passed vector is not large enough for the number of launched threads. Expected at least {launch_size} but got {}",
            self.len()
        );
        ThreadIndexedVec {
            ptr: self.as_mut_ptr(),
            phantom: PhantomData,
        }
    }
}

#[cfg(any(target_arch = "amdgpu", target_arch = "nvptx64"))]
fn thread_id() -> usize {
    use crate::intrinsics::*;
    let dispatch = crate::dispatch_ptr();

    // Compute size as ((z * dimY) + y) * dimX + x
    let mut id =
        workitem_id_z() as usize + dispatch.workgroup_size_z as usize * workgroup_id_z() as usize;
    id *= dispatch.grid_size_y as usize;
    id += workitem_id_y() as usize + dispatch.workgroup_size_y as usize * workgroup_id_y() as usize;
    id *= dispatch.grid_size_x as usize;
    id += workitem_id_x() as usize + dispatch.workgroup_size_x as usize * workgroup_id_x() as usize;
    id
}

impl<'a, T> ThreadIndexedVec<'a, T> {
    /// Constructs a `ThreadIndexedVec` from a raw base pointer.
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

    /// Returns the base pointer of the wrapped `Vec`.
    pub fn as_mut_base_ptr(&mut self) -> *mut T {
        self.ptr
    }

    #[cfg(any(target_arch = "amdgpu", target_arch = "nvptx64"))]
    pub fn get(&self) -> &T {
        unsafe { &*self.ptr.add(thread_id()) }
    }

    #[cfg(any(target_arch = "amdgpu", target_arch = "nvptx64"))]
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.ptr.add(thread_id()) }
    }
}
