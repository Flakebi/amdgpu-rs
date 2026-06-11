#![cfg_attr(feature = "gpu", no_std)]
#![cfg_attr(feature = "gpu", feature(abi_gpu_kernel))]

use gpu_kernel::kernel;

gpu_kernel::kernel_lib!();

/// This kernel adds numbers from two slices and writes the result into a third.
#[kernel]
#[allow(improper_ctypes_definitions, improper_gpu_kernel_arg)]
pub unsafe fn kernel<'a, 'b>(a: &'a [u32], b: &'b [u32], c: *mut u32) {
    use gpu_kernel::intrinsics::workgroup_id_x;
    use gpu_kernel::intrinsics::workitem_id_x;

    let id = workitem_id_x() as usize + 32 * workgroup_id_x() as usize;

    // Add two input numbers and store into output
    unsafe {
        let mut n = 0;
        for i in 0..100 {
            n += a[id] + b[id] + a[(id * i) % a.len()] - b[a.len().wrapping_sub(id * i) % a.len()];
        }
        *c.add(id) = n;
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {
    use gpu_kernel::LaunchConfig;
    use gpu_kernel::PrefetchMem;

    // TODO Move to amdgpu-rs
    // TODO Build with CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS='-Ctarget-cpu=gfx...' or set in ~/.cargo/config.toml
    // TODO Document everything and add readmes
    // TODO Test on Ubuntu podman
    // TODO Add feature to amdgpu-device-libs-build to remove winnow and other dependencies?
    // TODO Example that transfers memory to GPU before usage, expose as impl Into<DeviceMem<&T>>
    // TODO Test type that is not copy

    // Create two vectors a and b to add together
    let mut a = Vec::new();
    let mut b = Vec::new();
    for i in 0..3200_000 {
        a.push(i);
        b.push(i);
    }

    // Create vector c to hold the results
    let mut c = Vec::new();
    c.resize(a.len(), 0);

    // We pass CPU pointers to the kernel, which works fine, though is potentially slow.
    // hipMemoryAdvise can be used to improve this.
    // Only heap variables are shared on dedicated AMD GPUs, so this cannot be constant slices.

    unsafe {
        kernel.launch(
            LaunchConfig::new()
                .threads_per_workgroup([32, 1, 1])
                .workgroups([(a.len() / 32) as u32, 1, 1]),
            PrefetchMem::prefetch_incoherent(a.as_slice()),
            PrefetchMem::prefetch_incoherent(b.as_slice()),
            PrefetchMem::copy(c.as_slice()).as_ptr() as *mut _,
        );
    }
    std::mem::forget(a);
    std::mem::forget(b);
    std::mem::forget(c);

    // println!("Result: {:?}", &c[..100]);
}
