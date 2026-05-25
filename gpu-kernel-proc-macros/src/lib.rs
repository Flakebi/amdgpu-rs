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
            use gpu_kernel::prelude::*;

            println!("Hehe, replaced hello World!");
        }

        // CPU version

        // TODO Name, args, make pub when original is pub
        #[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
        pub fn test(launch_config: ::gpu_kernel::LaunchConfig, i: u32) {
            static TEST_KERNEL: std::sync::LazyLock<::gpu_kernel::Kernel> = std::sync::LazyLock::new(|| {
                unsafe {
                    let mut function: ::gpu_kernel::hip_runtime_sys::hipFunction_t = std::ptr::null_mut();
                    // TODO actual name
                    let kernel_name = std::ffi::CString::new("test_kernel").expect("Invalid kernel name");
                    let result = ::gpu_kernel::hip_runtime_sys::hipModuleGetFunction(&mut function, GPU_KERNEL_MODULE.0, kernel_name.as_ptr());
                    assert_eq!(result, ::gpu_kernel::hip_runtime_sys::hipError_t::hipSuccess, "Failed to find kernel {:?}", kernel_name);
                    // TODO
                    ::gpu_kernel::Kernel::new(function)
                }
            });

            // Launch kernel
            struct Args {
                i: u32,
            }
            let args: Args = Args { i };
            TEST_KERNEL.launch(launch_config, args);
        }
    };

    proc_macro::TokenStream::from(output)
}

#[proc_macro]
pub fn kernel_lib_impl(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // Compile gpu crate here
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

    let mut cargo = Command::new("cargo");
    cargo.args(&[
        "build",
        "--target",
        "amdgcn-amd-amdhsa",
        "--lib",
        "--release",
        "-Zbuild-std=core,alloc",
        // "--verbose",
        // TODO Only when it exists in Cargo.toml
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

    // TODO Pick names that do not conflict
    let kernel_path = kernel_path.display().to_string();
    let output = quote! {
        static GPU_KERNEL_MODULE_DATA: &[u8] = std::include_bytes!(#kernel_path);
        static GPU_KERNEL_MODULE: std::sync::LazyLock<::gpu_kernel::Module> = std::sync::LazyLock::new(|| {
            // Dummy include to mark the file as used and re-run the macro/build if it changed
            // Does it actually work like this?
            // TODO Add for all files in the crate
            let _ = std::include_bytes!("main.rs");
            let _ = std::include_bytes!("../Cargo.toml");


            unsafe {
                // TODO Not always 0
                let result = ::gpu_kernel::hip_runtime_sys::hipSetDevice(0);
                assert_eq!(result, ::gpu_kernel::hip_runtime_sys::hipError_t::hipSuccess);
                let mut module: ::gpu_kernel::hip_runtime_sys::hipModule_t = std::ptr::null_mut();
                let result =
                    ::gpu_kernel::hip_runtime_sys::hipModuleLoadData(&mut module, GPU_KERNEL_MODULE_DATA.as_ptr() as *const std::ffi::c_void);
                assert_eq!(result, ::gpu_kernel::hip_runtime_sys::hipError_t::hipSuccess);
                ::gpu_kernel::Module(module)
            }
        });
    };
    proc_macro::TokenStream::from(output)
}
