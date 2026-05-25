extern crate proc_macro;

use std::env;
use std::path::PathBuf;
use std::process::Command;

use quote::{format_ident, quote};
use syn::{FnArg, ItemFn, Pat, parse_macro_input};

#[proc_macro_attribute]
pub fn kernel(
    _attr: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let func = parse_macro_input!(input as ItemFn);
    let attrs = func.attrs;
    let vis = func.vis;
    let code = func.block;
    let orig_ident = func.sig.ident;
    let kernel_ident = format_ident!("{}_kernel", orig_ident);
    let inputs = func.sig.inputs;
    let output = func.sig.output;

    assert!(
        func.sig.asyncness.is_none(),
        "#[kernel] `{orig_ident}` cannot be async",
    );
    assert!(
        func.sig.unsafety.is_none(),
        "#[kernel] `{orig_ident}` cannot be unsafe",
    );
    assert!(
        func.sig.generics.lt_token.is_none(),
        "#[kernel] `{orig_ident}` cannot be generic",
    );
    assert!(
        func.sig.variadic.is_none(),
        "#[kernel] `{orig_ident}` cannot be variadic"
    );

    // For the argument list and struct def `type0`
    let mut input_tys = Vec::new();
    // For the struct initialization `arg0`
    let mut input_names = Vec::new();

    for (i, arg) in inputs.iter().enumerate() {
        let mut name = format_ident!("arg{i}");
        let ty;

        match arg {
            FnArg::Receiver(_) => {
                panic!("#[kernel] `{orig_ident}` cannot have a `self` argument");
            }
            FnArg::Typed(arg) => {
                assert!(
                    arg.attrs.is_empty(),
                    "#[kernel] `{orig_ident}` arg `{name}` cannot have attributes"
                );
                if let Pat::Ident(ident) = &*arg.pat {
                    name = ident.ident.clone();
                }
                ty = arg.ty.clone();
            }
        }
        input_tys.push(ty);
        input_names.push(name);
    }

    let output = quote! {
        // GPU code

        // Safety: Append "_kernel" to create a name that can use no_mangle
        #[cfg(any(target_arch = "amdgpu", target_arch = "nvptx64"))]
        #[unsafe(no_mangle)]
        #(#attrs)*
        #vis unsafe extern "gpu-kernel" fn #kernel_ident(#inputs) #output
            #code

        // CPU code

        #[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
        #(#attrs)*
        #vis fn #orig_ident(launch_config: ::gpu_kernel::LaunchConfig, #(#input_names: #input_tys),*) {
            static KERNEL: std::sync::LazyLock<::gpu_kernel::Kernel> = std::sync::LazyLock::new(|| {
                GPU_KERNEL_MODULE.get_kernel(stringify!(#kernel_ident))
            });

            // Assemble arguments
            #[repr(C)]
            struct Args {
                #(#input_names: #input_tys),*
            }
            let args: Args = Args { #(#input_names),* };
            // Launch kernel
            KERNEL.launch(launch_config, args);
        }
    };

    proc_macro::TokenStream::from(output)
}

#[proc_macro]
pub fn kernel_lib_impl(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // Compile gpu crate here
    let crate_name = env::var("CARGO_CRATE_NAME").expect("$CARGO_CRATE_NAME must be set");
    let kernel_file = format!("{crate_name}.elf");
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("$CARGO_MANIFEST_DIR must be set"));
    let manifest_path =
        PathBuf::from(env::var("CARGO_MANIFEST_PATH").expect("$CARGO_MANIFEST_PATH must be set"));
    // TODO Support CARGO_TARGET_DIR if set
    let kernel_path = manifest_dir
        .join("target")
        .join("amdgcn-amd-amdhsa")
        .join("release")
        .join(&kernel_file);

    // Custom settings
    let flags = env::var("GPU_CARGO_BUILD_RUSTFLAGS").unwrap_or_default();
    // Default to release build, set GPU_CARGO_BUILD_RELEASE=0 to build debug
    let build_release = env::var("GPU_CARGO_BUILD_RELEASE").unwrap_or_default() != "0";
    let build_verbose = env::var("GPU_CARGO_BUILD_VERBOSE").unwrap_or_default() == "1";
    let target = {
        #[cfg(feature = "amd")]
        "amdgcn-amd-amdhsa"
    };

    let mut cargo = Command::new("cargo");
    cargo.args(&[
        "build",
        "--target",
        target,
        "--lib",
        "-Zbuild-std=core,alloc",
        // TODO Only when it exists in Cargo.toml
        "--features",
        "gpu",
    ]);
    if build_release {
        cargo.arg("--release");
    }
    if build_verbose {
        cargo.arg("--verbose");
    }
    // TODO If we have a reliable way to find the target-cpu, we can call amdgpu-device-libs-build here
    // Search in env build flags and .cargo/config.toml?
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

    let kernel_path = kernel_path.display().to_string();
    let manifest_path = manifest_path.display().to_string();
    let output = quote! {
        // TODO Use proc_macro_tracked_path
        // Changes to Cargo.toml can affect the GPU build.
        // Dummy include to re-run the macro if it changed.
        const _: &[u8] = std::include_bytes!(#manifest_path);
        const _: std::option::Option<&str> = std::option_env!("GPU_CARGO_BUILD_RUSTFLAGS");
        const _: std::option::Option<&str> = std::option_env!("GPU_CARGO_BUILD_RELEASE");
        const _: std::option::Option<&str> = std::option_env!("GPU_CARGO_BUILD_VERBOSE");

        static GPU_KERNEL_MODULE_DATA: &[u8] = std::include_bytes!(#kernel_path);
        static GPU_KERNEL_MODULE: std::sync::LazyLock<::gpu_kernel::Module> = std::sync::LazyLock::new(|| ::gpu_kernel::Module::new(GPU_KERNEL_MODULE_DATA));
    };
    proc_macro::TokenStream::from(output)
}
