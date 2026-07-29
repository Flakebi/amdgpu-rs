use std::ffi::CString;
use std::path::PathBuf;

use clap::Parser;
use hip_runtime_sys::{
    hipDeviceProp_t, hipDeviceSynchronize, hipDriverGetVersion, hipError_t, hipFunction_t,
    hipGetDevice, hipGetDeviceCount, hipGetDeviceProperties, hipInit, hipModule_t,
    hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoadData, hipModuleUnload,
    hipRuntimeGetVersion, hipSetDevice,
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

/// Function on the host that will be called from the GPU.
///
/// We get output and input as pointers.
/// `output` is 16 byte (2x u64), input is 7x u64
unsafe extern "C" fn host_func(output: *mut u64, input: *const u64) {
    unsafe {
        // Read first of 7 arguments
        let num = *input;
        println!("Function on the host, got {num}");
        *output = num + 42;
        // Output consists of two u64, zero out the second one
        *output.add(1) = 0;
    }
}

fn main() {
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
        let (name, arch) = unsafe {
            let mut device_prop: hipDeviceProp_t = std::mem::zeroed();
            let result = hipGetDeviceProperties(&mut device_prop, i);
            assert_eq!(result, hipError_t::hipSuccess);
            (
                get_str(&device_prop.name).to_string(),
                get_str(&device_prop.gcnArchName).to_string(),
            )
        };
        println!("Device {i}: {name} ({arch})");
    }

    unsafe {
        println!("Set device {}", args.device_index);
        let result = hipSetDevice(args.device_index);
        assert_eq!(result, hipError_t::hipSuccess);
        let mut device = 0;
        let result = hipGetDevice(&mut device);
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
        // Pass the address of host_func
        let kernel_args: &mut [*mut std::ffi::c_void] =
            &mut [host_func as *const std::ffi::c_void as *mut _];
        let mut size = std::mem::size_of_val(kernel_args);
        println!("host_func has address {:?}", host_func as *const ());

        #[allow(clippy::manual_dangling_ptr)]
        let mut config = [
            0x1 as *mut std::ffi::c_void,                   // Next come arguments
            kernel_args as *mut _ as *mut std::ffi::c_void, // Pointer to arguments
            0x2 as *mut std::ffi::c_void,                   // Next comes size
            std::ptr::addr_of_mut!(size) as *mut std::ffi::c_void, // Pointer to size of arguments
            0x3 as *mut std::ffi::c_void,                   // End
        ];

        // Launch two workgroups (2x1x1), each of the size 32x1x1
        println!("Launch {}", args.kernel);
        let result = hipModuleLaunchKernel(
            function,
            2,                    // Workgroup count x
            1,                    // Workgroup count y
            1,                    // Workgroup count z
            32,                   // Workgroup dim x
            1,                    // Workgroup dim y
            1,                    // Workgroup dim z
            0,                    // sharedMemBytes for extern shared variables
            std::ptr::null_mut(), // stream
            std::ptr::null_mut(), // params (unimplemented in hip)
            config.as_mut_ptr(),  // arguments
        );
        assert_eq!(result, hipError_t::hipSuccess);

        println!("Wait for finish");
        let result = hipDeviceSynchronize();
        assert_eq!(result, hipError_t::hipSuccess);

        println!("Free");
        let result = hipModuleUnload(module);
        assert_eq!(result, hipError_t::hipSuccess);

        println!("Finished");
    }
}
