use std::ffi::CString;
use std::path::PathBuf;

use clap::Parser;
use hip_runtime_sys::{
    hipDeviceProp_t, hipDeviceSynchronize, hipDeviceptr_t, hipDriverGetVersion, hipError_t,
    hipFree, hipFunction_t, hipGetDevice, hipGetDeviceCount, hipGetDeviceProperties, hipInit,
    hipMalloc, hipMemcpyDtoH, hipMemcpyHtoD, hipModule_t, hipModuleGetFunction,
    hipModuleLaunchKernel, hipModuleLoadData, hipModuleUnload, hipRuntimeGetVersion, hipSetDevice,
};

#[derive(Parser)]
#[command()]
struct Cli {
    /// Path to compiled GPU kernel
    file: PathBuf,

    /// Index of the device to use
    #[arg(short, long, default_value_t = 0)]
    device_index: i32,

    /// Name of the kernel
    #[arg(short, long, default_value = "kernel")]
    kernel: String,
}

fn get_str(s: &[i8]) -> &str {
    let cs =
        std::ffi::CStr::from_bytes_until_nul(unsafe { std::mem::transmute::<&[i8], &[u8]>(s) })
            .unwrap();
    cs.to_str().unwrap()
}

fn main() -> std::process::ExitCode {
    // Parse arguments
    let args = Cli::parse();

    // Get some system information from HIP
    // Adjusted from https://github.com/cjordan/hip-sys/blob/5a55ab891dec0446a6b09152c385b1c8e4e6df45/examples/hip_info.rs
    // under MIT/Apache 2.0 by Dev Null
    let result = unsafe { hipInit(0) };
    assert_eq!(result, hipError_t::hipSuccess);

    let mut driver_version: i32 = 0;
    let result = unsafe { hipDriverGetVersion(&mut driver_version) };
    assert_eq!(result, hipError_t::hipSuccess);
    println!("Driver Version: {driver_version}");

    let mut runtime_version: i32 = 0;
    let result = unsafe { hipRuntimeGetVersion(&mut runtime_version) };
    assert_eq!(result, hipError_t::hipSuccess);
    println!("Runtime Version: {runtime_version}");

    // Get devices on the system and some of their information
    let mut device_count: i32 = 0;
    let result = unsafe { hipGetDeviceCount(&mut device_count) };
    assert_eq!(result, hipError_t::hipSuccess);
    println!("Device Count: {device_count}");

    for i in 0..device_count {
        // `arch` is the gfx version for which kernels need to be compiled
        let (name, arch, device_prop) = unsafe {
            let mut device_prop: hipDeviceProp_t = std::mem::zeroed();
            let result = hipGetDeviceProperties(&mut device_prop, i);
            assert_eq!(result, hipError_t::hipSuccess);
            (
                get_str(&device_prop.name).to_string(),
                get_str(&device_prop.gcnArchName).to_string(),
                device_prop,
            )
        };
        println!("Device {i}");
        println!("  {name} ({arch}) | multi {}", device_prop.isMultiGpuBoard);
        println!(
            "  mem    | VRAM: {}GiB, shared/block: {}KiB, ",
            device_prop.totalGlobalMem / (1024 * 1024 * 1024),
            device_prop.sharedMemPerBlock / 1024
        );
        println!(
            "  thread | max/block: {}, warpSize {}, {} processors with {} max threads, max [{} {} {}]",
            device_prop.maxThreadsPerBlock,
            device_prop.warpSize,
            device_prop.multiProcessorCount,
            device_prop.maxThreadsPerMultiProcessor,
            device_prop.maxThreadsDim[0],
            device_prop.maxThreadsDim[1],
            device_prop.maxThreadsDim[2]
        );
        println!(
            "  grid   | max [{} {} {}]",
            device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]
        );
    }

    const LEN: usize = 1024;
    let mut a: Vec<u8> = vec![0; LEN];
    let mut b: Vec<u8> = vec![0; LEN];

    for (i, elem) in a.iter_mut().enumerate() {
        *elem = i as u8;
    }

    unsafe {
        println!("Set device {}", args.device_index);
        let result = hipSetDevice(args.device_index);
        assert_eq!(result, hipError_t::hipSuccess);
        let mut device = 0;
        let result = hipGetDevice(&mut device);
        assert_eq!(result, hipError_t::hipSuccess);

        // Allocate two buffers on the GPU
        println!("Alloc memory");
        let mut ad: hipDeviceptr_t = std::ptr::null_mut();
        let mut bd: hipDeviceptr_t = std::ptr::null_mut();
        let result = hipMalloc(&mut ad, LEN);
        assert_eq!(result, hipError_t::hipSuccess);
        let result = hipMalloc(&mut bd, LEN);
        assert_eq!(result, hipError_t::hipSuccess);

        // Copy a and b to GPU buffers
        println!("Copy memory from {:?} (cpu) to {:?} (gpu)", a.as_ptr(), ad);
        let result = hipMemcpyHtoD(ad, a.as_mut_ptr() as *mut std::ffi::c_void, LEN);
        assert_eq!(result, hipError_t::hipSuccess);
        println!("Copy memory from {:?} (cpu) to {:?} (gpu)", b.as_ptr(), bd);
        let result = hipMemcpyHtoD(bd, b.as_mut_ptr() as *mut std::ffi::c_void, LEN);
        assert_eq!(result, hipError_t::hipSuccess);

        // Load the executable that was compiled for the GPU
        println!("Load module from {}", args.file.display());
        let module_data = std::fs::read(args.file).unwrap();
        let mut module: hipModule_t = std::ptr::null_mut();
        let result =
            hipModuleLoadData(&mut module, module_data.as_ptr() as *const std::ffi::c_void);
        assert_eq!(result, hipError_t::hipSuccess);

        // Get kernel function from loaded module
        println!("Get function {}", args.kernel);
        let mut function: hipFunction_t = std::ptr::null_mut();
        let kernel_name = CString::new(args.kernel.clone()).expect("Invalid kernel name");
        let result = hipModuleGetFunction(&mut function, module, kernel_name.as_ptr());
        assert_eq!(result, hipError_t::hipSuccess);

        // Assemble arguments for the kernel.
        // Pass two pointers, ad and bd
        let kernel_args: &mut [*mut std::ffi::c_void] = &mut [ad, bd];
        let mut size = std::mem::size_of_val(kernel_args);

        #[allow(clippy::manual_dangling_ptr)]
        let mut config = [
            0x1 as *mut std::ffi::c_void,                   // Next come arguments
            kernel_args as *mut _ as *mut std::ffi::c_void, // Pointer to arguments
            0x2 as *mut std::ffi::c_void,                   // Next comes size
            std::ptr::addr_of_mut!(size) as *mut std::ffi::c_void, // Pointer to size of arguments
            0x3 as *mut std::ffi::c_void,                   // End
        ];

        // Launch two workgroups (2x1x1), each of the size (LEN/2)x1x1
        println!("Launch {}", args.kernel);
        let result = hipModuleLaunchKernel(
            function,
            2,                    // Workgroup count x
            1,                    // Workgroup count y
            1,                    // Workgroup count z
            LEN as u32 / 2,       // Workgroup dim x
            1,                    // Workgroup dim y
            1,                    // Workgroup dim z
            LEN as u32 / 2,       // sharedMemBytes for extern shared variables
            std::ptr::null_mut(), // stream
            std::ptr::null_mut(), // params (unimplemented in hip)
            config.as_mut_ptr(),  // arguments
        );
        assert_eq!(result, hipError_t::hipSuccess);

        println!("Wait for finish");
        let result = hipDeviceSynchronize();
        assert_eq!(result, hipError_t::hipSuccess);

        println!("Copy memory back");
        let result = hipMemcpyDtoH(b.as_mut_ptr() as *mut std::ffi::c_void, bd, LEN);
        assert_eq!(result, hipError_t::hipSuccess);

        // Print first part of result buffer
        println!("Output: {:02x?}", &b[..32]);

        // Check if b contains now the same as a
        let mismatch_count = a.iter().zip(b.iter()).filter(|(a, b)| a != b).count();

        let mut success = true;
        if mismatch_count == 0 {
            println!("PASSED!");
        } else if mismatch_count <= LEN / 2 && a[..LEN / 2] == b[..LEN / 2] {
            println!("Failed for the second half! (This is expected for the simple vector_copy)");
        } else {
            println!("FAILED!");
            success = false;
        }

        println!("Free");
        let result = hipModuleUnload(module);
        assert_eq!(result, hipError_t::hipSuccess);
        let result = hipFree(ad);
        assert_eq!(result, hipError_t::hipSuccess);
        let result = hipFree(bd);
        assert_eq!(result, hipError_t::hipSuccess);

        println!("Finished");
        if success {
            std::process::ExitCode::SUCCESS
        } else {
            std::process::ExitCode::FAILURE
        }
    }
}
