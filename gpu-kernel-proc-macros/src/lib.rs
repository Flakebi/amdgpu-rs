extern crate proc_macro;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

use quote::{format_ident, quote};
use syn::{FnArg, ItemFn, Pat, parse_macro_input};
use toml::Table;
use toml::map::Entry;
use toml::value::Array;

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
    let generics = func.sig.generics;
    let output = func.sig.output;

    assert!(
        func.sig.asyncness.is_none(),
        "#[kernel] `{orig_ident}` cannot be async",
    );
    // TODO Force kernel to be unsafe for now because &mut is never allowed
    assert!(
        func.sig.unsafety.is_none(),
        "#[kernel] `{orig_ident}` cannot be unsafe",
    );
    // TODO Forbid only type generics
    /*assert!(
        func.sig.generics.lt_token.is_none(),
        "#[kernel] `{orig_ident}` cannot be generic",
    );*/
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
        #vis unsafe extern "gpu-kernel" fn #kernel_ident #generics(#inputs) #output
            #code

        // CPU code

        #[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
        #(#attrs)*
        #vis fn #orig_ident #generics(launch_config: ::gpu_kernel::LaunchConfig, #(#input_names: #input_tys),*) {
            static KERNEL: std::sync::LazyLock<::gpu_kernel::Kernel> = std::sync::LazyLock::new(|| {
                GPU_KERNEL_MODULE.get_kernel(stringify!(#kernel_ident))
            });

            // Assemble arguments
            #[repr(C)]
            struct Args #generics {
                #(#input_names: #input_tys),*
            }
            let mut args: Args = Args { #(#input_names),* };
            // Launch kernel
            KERNEL.launch(launch_config, &mut args);
        }
    };

    proc_macro::TokenStream::from(output)
}

// TODO Split into smaller functions
#[proc_macro]
pub fn kernel_lib_impl(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let target = {
        #[cfg(feature = "amd")]
        "amdgcn-amd-amdhsa"
    };
    let target_env = target.replace('-', "_").to_uppercase();
    let target_rustflags = format!("CARGO_TARGET_{target_env}_RUSTFLAGS");
    let target_cargoflags = format!("CARGO_TARGET_{target_env}_FLAGS");

    // Compile gpu crate here
    let crate_name = env::var("CARGO_CRATE_NAME").expect("$CARGO_CRATE_NAME must be set");
    let kernel_file = format!("{crate_name}.elf");
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("$CARGO_MANIFEST_DIR must be set"));
    let manifest_path =
        PathBuf::from(env::var("CARGO_MANIFEST_PATH").expect("$CARGO_MANIFEST_PATH must be set"));
    let lock_path = manifest_dir.join("Cargo.lock");
    // Use CARGO_TARGET_DIR if set
    let target_dir = env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest_dir.join("target"));
    let kernel_path = target_dir
        .join("amdgcn-amd-amdhsa")
        .join("release")
        .join(&kernel_file);

    let env_rustflags = env::var(&target_rustflags).unwrap_or_default();
    // Custom setting, defaults to --release
    let cargoflags = env::var(&target_cargoflags).unwrap_or_else(|_| "--release".into());

    // Get rustflags from env and .cargo/config.toml
    let cargo_config_path = manifest_dir.join(".cargo").join("config.toml");
    let config_rustflags =
        if fs::exists(&cargo_config_path).expect("Failed to check for .cargo/config.toml") {
            let config =
                fs::read_to_string(&cargo_config_path).expect("Failed to read .cargo/config.toml");
            let config = config
                .parse::<Table>()
                .expect("Invalid toml in .cargo/config.toml");
            config
                .get("target")
                .and_then(|v| {
                    v.as_table()
                        .expect("Failed to parse .cargo/config.toml")
                        .get(target)
                })
                .and_then(|v| {
                    v.as_table()
                        .expect("Failed to parse .cargo/config.toml")
                        .get("rustflags")
                })
                .map(|v| {
                    v.as_array()
                        .expect("Failed to parse .cargo/config.toml")
                        .iter()
                        .map(|v| {
                            v.as_str()
                                .expect("Failed to parse .cargo/config.toml")
                                .to_string()
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };
    let all_rustflags = format!("{env_rustflags} {}", config_rustflags.join(" "));

    // Find important things in flags
    let target_cpu = {
        let i = all_rustflags.rfind("target-cpu").unwrap_or_else(|| panic!("Did not find target-cpu, make sure to set `-Ctarget-cpu=...` in ${target_rustflags}"));
        let start = i + "target-cpu".len() + 1;
        let end = all_rustflags[start..]
            .find(' ')
            .map(|i| start + i)
            .unwrap_or(all_rustflags.len());
        &all_rustflags[start..end]
    };
    // Enabled and not disabled or enabling comes later than disabling
    let is_wave64_enabled = all_rustflags
        .rfind("+wavefrontsize64")
        .map(|i| {
            if let Some(j) = all_rustflags.rfind("-wavefrontsize64") {
                i > j
            } else {
                true
            }
        })
        .unwrap_or_default();

    let link_args =
        amdgpu_device_libs_build::get_link_args(is_wave64_enabled, &target_cpu).link_args;
    let new_rustflags = link_args
        .iter()
        .map(|v| format!("-Clink-arg={v}"))
        .collect::<Vec<_>>();

    // Copy Cargo.toml, insert lib.path = main.rs if lib does not exist, set lib.crate-type = cdylib
    let cargo_toml = fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {e}", manifest_path.display()));
    let mut cargo_toml = cargo_toml
        .parse::<Table>()
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", manifest_path.display()));
    let has_gpu_feature = cargo_toml
        .get("features")
        .map(|v| {
            v.as_table()
                .expect("features needs to be a toml table")
                .contains_key("gpu")
        })
        .unwrap_or_default();
    let has_lib = cargo_toml.contains_key("lib")
        || fs::exists(manifest_dir.join("src").join("lib.rs")).expect("Failed to check for lib.rs");
    let lib_config = cargo_toml
        .entry("lib")
        .or_insert_with(|| Table::new().into())
        .as_table_mut()
        .expect("lib needs to be a toml table");

    // Set or fixup lib path
    match lib_config.entry("path") {
        Entry::Vacant(e) => {
            let path;
            if has_lib {
                path = "../../src/lib.rs";
            } else {
                path = "../../src/main.rs";
            }
            e.insert(path.into());
        }
        Entry::Occupied(mut e) => {
            // Fixup relative path
            let path = Path::new(e.get().as_str().expect("lib path must be a toml string"));
            if path.is_relative() {
                let new = Path::new("../..").join(path).display().to_string();
                e.insert(new.into());
            }
        }
    }

    let mut a = Array::new();
    a.push("cdylib".into());
    lib_config.insert("crate-type".into(), a.into());

    // Fixup all relative dependency paths in the Cargo.toml
    let fix_dep = |v: &mut toml::Value| {
        if let Some(v) = v.as_table_mut() {
            if let Some(p) = v.get_mut("path") {
                let path = Path::new(p.as_str().expect("Dependency path must be a toml string"));
                if path.is_relative() {
                    let new = Path::new("../..").join(path).display().to_string();
                    *p = new.into();
                }
            }
        }
    };
    let dep_keys = &["dependencies", "build-dependencies", "dev-dependencies"];
    let fix_all_deps = |t: &mut Table| {
        for k in dep_keys {
            if let Some(t) = t.get_mut(*k) {
                let t = t
                    .as_table_mut()
                    .unwrap_or_else(|| panic!("{k} must be a toml table"));
                for (_, v) in t.iter_mut() {
                    fix_dep(v);
                }
            }
        }
    };

    // Either in root or in target.<something>
    fix_all_deps(&mut cargo_toml);
    if let Some(t) = cargo_toml.get_mut("target") {
        let t = t.as_table_mut().expect("target must be a toml table");
        for (_, v) in t.iter_mut() {
            fix_all_deps(v.as_table_mut().expect("target must contain toml tables"));
        }
    }

    // Write new Cargo.toml
    let gpu_toml_dir = target_dir.join("gpu-kernel");
    fs::create_dir_all(&gpu_toml_dir).expect("Failed to create gpu-kernel target dir");
    let gpu_toml = gpu_toml_dir.join("Cargo.toml");
    fs::write(&gpu_toml, cargo_toml.to_string().as_bytes())
        .expect("Failed to write GPU Cargo.toml");
    // Copy Cargo.lock
    if let Err(e) = fs::copy(&lock_path, gpu_toml_dir.join("Cargo.lock")) {
        println!("Warning: Failed to copy Cargo.lock to GPU directory ({e}), ignoring");
    }

    let mut cargo = Command::new("cargo");
    cargo.args(&[
        "build",
        "--target",
        target,
        "--lib",
        "-Zbuild-std=core,alloc",
        "-m",
        &gpu_toml.display().to_string(),
        "--target-dir",
        &target_dir.display().to_string(),
    ]);
    if has_gpu_feature {
        cargo.args(&["--features", "gpu"]);
    }
    for f in cargoflags.split(' ') {
        cargo.arg(f);
    }

    cargo.env(
        &target_rustflags,
        format!(
            "{env_rustflags} {} -Clinker-plugin-lto",
            new_rustflags.join(" ")
        ),
    );
    let res = cargo
        .status()
        .expect("Failed to run cargo to compile for GPU");
    if !res.success() {
        panic!("Cargo did not exit successfully, failed to compile for GPU");
    }

    let kernel_path = kernel_path.display().to_string();
    let manifest_path = manifest_path.display().to_string();
    let lock_path = lock_path.display().to_string();
    let output = quote! {
        // Changes to Cargo.toml can affect the GPU build.
        // Dummy include to re-run the macro if it changed.
        // Use proc_macro tracked path and env once it is stable.
        const _: &[u8] = std::include_bytes!(#manifest_path);
        const _: &[u8] = std::include_bytes!(#lock_path);
        const _: std::option::Option<&str> = std::option_env!("CARGO_TARGET_DIR");
        const _: std::option::Option<&str> = std::option_env!(#target_rustflags);
        const _: std::option::Option<&str> = std::option_env!(#target_cargoflags);

        static GPU_KERNEL_MODULE_DATA: &[u8] = std::include_bytes!(#kernel_path);
        static GPU_KERNEL_MODULE: std::sync::LazyLock<::gpu_kernel::Module> = std::sync::LazyLock::new(|| ::gpu_kernel::Module::new(GPU_KERNEL_MODULE_DATA));
    };
    proc_macro::TokenStream::from(output)
}
