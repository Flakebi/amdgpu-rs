extern crate proc_macro;

use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

use quote::quote;
use syn::{DeriveInput, parse_macro_input};

#[proc_macro_attribute]
pub fn kernel(
    _attr: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    // let input = proc_macro2::TokenStream::from(input);
    // let macro_input = parse_macro_input!(input as DeriveInput);
    // let input = parse_macro_input!(tokens as MyMacroInput);

    // let output: proc_macro2::TokenStream = {
    /* transform input */
    // };
    // let output: proc_macro2::TokenStream = {
    let output = quote! {
        // GPU version

        // Safety: Append "_kernel" to create a name that can use no_mangle
        // TODO Name, args, make pub when original is pub
        #[cfg(any(target_arch = "amdgpu", target_arch = "nvptx64"))]
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn test_kernel(i: i32) {
            println!("Hehe, replaced hello World!");
        }

        // CPU version

        // TODO Name, args, make pub when original is pub
        #[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
        pub fn test(launch_config: ::gpu_kernel::LaunchConfig, i: u32) {
            // static TEST_KERNEL: std::sync::LazyLock<::gpu_kernel::Kernel> = std::sync::LazyLock::new(|| {
            unsafe {
                let result = ::gpu_kernel::hip_runtime_sys::hipSetDevice(0);
                assert_eq!(result, ::gpu_kernel::hip_runtime_sys::hipError_t::hipSuccess, "Failed to set device");
                let mut module: ::gpu_kernel::hip_runtime_sys::hipModule_t = std::ptr::null_mut();
                let result =
                    ::gpu_kernel::hip_runtime_sys::hipModuleLoadData(&mut module, MODULE_DATA.as_ptr() as *const std::ffi::c_void);
                assert_eq!(result, ::gpu_kernel::hip_runtime_sys::hipError_t::hipSuccess, "Failed to load GPU module");

                let mut function: ::gpu_kernel::hip_runtime_sys::hipFunction_t = std::ptr::null_mut();
                let kernel_name = std::ffi::CString::new("test_kernel").expect("Invalid kernel name");
                let result = ::gpu_kernel::hip_runtime_sys::hipModuleGetFunction(&mut function, module, kernel_name.as_ptr());
                assert_eq!(result, ::gpu_kernel::hip_runtime_sys::hipError_t::hipSuccess, "Failed to find kernel {:?}", kernel_name);
                // TODO
                // ::gpu_kernel::Kernel { id: 1 }
            // });

                // Launch kernel
                struct Args {
                    i: u32,
                }
                let kernel_args: &mut Args = &mut Args { i };
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
                let result = ::gpu_kernel::hip_runtime_sys::hipModuleLaunchKernel(
                    function,
                    launch_config.workgroups[0],
                    launch_config.workgroups[1],
                    launch_config.workgroups[2],
                    launch_config.threads_per_workgroups[0],
                    launch_config.threads_per_workgroups[1],
                    launch_config.threads_per_workgroups[2],
                    0,                    // sharedMemBytes for extern shared variables
                    std::ptr::null_mut(), // stream
                    std::ptr::null_mut(), // params (unimplemented in hip)
                    config.as_mut_ptr(),  // arguments
                );
                assert_eq!(result, ::gpu_kernel::hip_runtime_sys::hipError_t::hipSuccess, "Failed to launch kernel");

                let result = ::gpu_kernel::hip_runtime_sys::hipDeviceSynchronize();
                assert_eq!(result, ::gpu_kernel::hip_runtime_sys::hipError_t::hipSuccess, "Failed to wait for kernel to finish");
            }
        }
    };

    proc_macro::TokenStream::from(output)
}

#[proc_macro]
pub fn kernel_lib_impl(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // Compile gpu crate here
    // TODO Do nothing when compiling for gpu
    let crate_name = env::var("CARGO_CRATE_NAME").expect("$CARGO_CRATE_NAME must be set");
    let kernel_file = format!("{crate_name}.elf"); // TODO Also for nvptx?
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("$CARGO_MANIFEST_DIR must be set"));
    let target_dir = manifest_dir.join("target").join("gpu-kernel");
    let kernel_path = target_dir
        .join("target")
        .join("amdgcn-amd-amdhsa")
        .join("release")
        .join(&kernel_file);
    fs::create_dir_all(&target_dir).expect("Failed to create build dir");

    // TODO Debug when requested

    // Add #![no_std] to the lib
    // TODO Remove, incompatible with #![...] in the real lib.rs
    let new_lib_path = target_dir.join("lib.rs");
    fs::write(&new_lib_path, "#![no_std] include!(\"../../src/lib.rs\")")
        .expect("Failed to write dummy lib.rs");

    let new_cargo_toml = target_dir.join("Cargo.toml");
    fs::write(
        &new_cargo_toml,
        r#"
[package]
name = "vector_add"
version = "0.1.0"
edition = "2024"
build = "../../build.rs"

[lib]
crate-type = ["cdylib"]
path = "../../src/lib.rs"

[features]
gpu = []

[build-dependencies]
amdgpu-device-libs-build = "0.1"

[dependencies]
gpu-kernel = { path = "../../../.." }
amdgpu-device-libs = "0.1"
        "#,
    )
    .expect("Failed to write dummy Cargo.toml");

    let mut cargo = Command::new("cargo");
    cargo.args(&[
        "build",
        "-m",
        &format!("{}", new_cargo_toml.display()),
        "--target",
        "amdgcn-amd-amdhsa",
        // Use different target dir, so the main cargo does not block the build dir?
        // "--target-dir",
        // "target/gpu-kernel",
        "--lib",
        // TODO Copy Cargo.lock?
        // "--offline",
        "--release",
        "-Zbuild-std=core,alloc",
        // "--verbose",
        "--features",
        "gpu",
    ]);
    let flags = env::var("GPU_CARGO_BUILD_RUSTFLAGS").unwrap_or_default();
    cargo.env(
        "CARGO_BUILD_RUSTFLAGS",
        // format!("{flags} --crate-type cdylib -Clinker-plugin-lto"),
        format!("{flags} --verbose -Clinker-plugin-lto"),
    );
    let res = cargo
        .status()
        .expect("Failed to run cargo to compile for GPU");
    if !res.success() {
        panic!("Cargo did not exit successfully, failed to compile for GPU");
    }

    // TODO Copy Cargo.toml and add cdylib for gpu

    // TODO Pick names that do not conflict
    let kernel_path = kernel_path.display().to_string();
    let output = quote! {
        static MODULE_DATA: &[u8] = std::include_bytes!(#kernel_path);
        /*static MODULE: ::gpu_kernel::hip_runtime_sys::hipModule_t = {
            // Dummy include to mark the file as used
            // TODO Add for all files in the crate
            let _ = std::include_bytes!("lib.rs");
            let _ = std::include_bytes!("../Cargo.toml");


            unsafe {
                // TODO Not always 0
                let result = ::gpu_kernel::hip_runtime_sys::hipSetDevice(0);
                assert_eq!(result, ::gpu_kernel::hip_runtime_sys::hipError_t::hipSuccess);
                let mut module: ::gpu_kernel::hip_runtime_sys::hipModule_t = std::ptr::null_mut();
                let result =
                    ::gpu_kernel::hip_runtime_sys::hipModuleLoadData(&mut module, MODULE_DATA.as_ptr() as *const std::ffi::c_void);
                assert_eq!(result, ::gpu_kernel::hip_runtime_sys::hipError_t::hipSuccess);
            }
        };*/
    };
    proc_macro::TokenStream::from(output)
}
