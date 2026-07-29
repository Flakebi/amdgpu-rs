{
  description = "Examples and support libraries for the amdgpu Rust target";

  inputs = {
    crane.url = "github:ipetkov/crane";
    fenix.url = "github:nix-community/fenix";
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    fenix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      crane,
      fenix,
      flake-utils,
    }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (
      system:
      let
        pkgs = (import nixpkgs) {
          inherit system;
        };
        lib = pkgs.lib;

        native-toolchain =
          with fenix.packages.${system};
          combine [
            complete.rustc
            complete.rust-src
            complete.cargo
            complete.clippy
            complete.rustfmt
          ];

        craneLib = (crane.mkLib pkgs).overrideToolchain native-toolchain;

        cpuTy = "cpu";
        gpuTy = "gpu";
        gpuKernelTy = "gpu-kernel";

        # Common arguments can be set here to avoid repeating them later
        craneArgs =
          ty: path:
          let
            # Do not use craneLib.cleanCargoSource, otherwise it does not find util32.bc
            src = if ty != cpuTy then ./. else craneLib.cleanCargoSource ./.;
            cargoLock = ./${path}/Cargo.lock;
          in
          {
            inherit src cargoLock;
            cargoToml = ./${path}/Cargo.toml;
            postUnpack = ''
              cd $sourceRoot/${path}
              sourceRoot="."
            '';
            strictDeps = true;
            doCheck = ty == cpuTy;

            cargoVendorDir = craneLib.vendorMultipleCargoDeps {
              inherit (craneLib.findCargoFiles src) cargoConfigs;
              cargoLockList = [
                cargoLock

                "${native-toolchain}/lib/rustlib/src/rust/library/Cargo.lock"
              ];
            };

            # TODO Remove
            ROCM_PATH = "${pkgs.rocmPackages.clr}";
            ROCM_DEVICE_LIB_PATH = "${pkgs.rocmPackages.rocm-device-libs}";
            CARGO_BUILD_RUSTFLAGS = "--deny warnings";
          }
          // (
            if ty == cpuTy then
              { }
            else if ty == gpuTy then
              {
                CARGO_BUILD_RUSTFLAGS = "--deny warnings -Ctarget-cpu=gfx1036";
              }
            else
              {
                CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS = "-Ctarget-cpu=gfx1036";
              }
          );

        cpu_pkgs = [
          "amdgpu-device-libs-build"
          "gpu-kernel"
          "gpu-kernel-proc-macros"
          "examples-raw/default-cpu"
          "examples-raw/hostcall-cpu"
        ];
        gpu_pkgs = [
          "amdgpu-device-libs"
          "examples-raw/hostcall-gpu"
          "examples-raw/panic"
          "examples-raw/println"
          "examples-raw/vector_copy"
        ];
        gpu_kernel_pkgs = [
          "examples/hello_world"
          "examples/split_lib"
          "examples/vector_add"
          "examples/vector_add_fast"
        ];

        package_args =
          (lib.genAttrs cpu_pkgs (pkg: craneArgs cpuTy pkg))
          // (lib.genAttrs gpu_pkgs (pkg: craneArgs gpuTy pkg))
          // (lib.genAttrs gpu_kernel_pkgs (pkg: craneArgs gpuKernelTy pkg));

        # Build the actual crate itself, reusing the dependency artifacts.
        packages = builtins.mapAttrs (
          pkg: args: craneLib.buildPackage (args // { cargoArtifacts = craneLib.buildDepsOnly args; })
        ) package_args;

        packages_rustfmt = builtins.listToAttrs (
          builtins.map (pkg: {
            name = "${pkg}-fmt";
            value = craneLib.cargoFmt { src = craneLib.cleanCargoSource ./${pkg}; };
          }) (cpu_pkgs ++ gpu_pkgs ++ gpu_kernel_pkgs)
        );

        packages_clippy = builtins.mapAttrs (
          pkg: args:
          craneLib.cargoClippy (
            args
            // {
              cargoArtifacts = craneLib.buildDepsOnly args;
              cargoClippyExtraArgs =
                (lib.optionalString (builtins.elem pkg cpu_pkgs) "--all-targets ") + "-- --deny warnings";
            }
          )
        ) package_args;

        # Run all examples (except panic)
        runExamples =
          let
            default-cpu = lib.getExe' packages."examples-raw/default-cpu" "default-cpu";
            hostcall-cpu = lib.getExe' packages."examples-raw/hostcall-cpu" "hostcall-cpu";
          in
          pkgs.writeShellScriptBin "runExamples" ''
            set -euxo pipefail
            ${default-cpu} ${packages."examples-raw/vector_copy"}/lib/vector_copy.elf "$@"
            ${default-cpu} ${packages."examples-raw/println"}/lib/println.elf "$@"
            ${hostcall-cpu} ${packages."examples-raw/hostcall-gpu"}/lib/hostcall_gpu.elf "$@"

            ${lib.getExe' packages."examples/hello_world" "hello_world"}
            ${lib.getExe' packages."examples/split_lib" "split_lib"}
            ${lib.getExe' packages."examples/vector_add" "vector_add"}
            ${lib.getExe' packages."examples/vector_add_fast" "vector_add_fast"}
          '';
      in
      {
        packages = packages // {
          inherit runExamples;
          inherit native-toolchain;
        };

        apps = {
          runExamples = flake-utils.lib.mkApp { drv = runExamples; };
        };

        checks = {
          typos = pkgs.runCommand "check-typos" { } ''
            ${pkgs.typos}/bin/typos ${self}
            mkdir -p $out
          '';

          toml-format = craneLib.taploFmt {
            pname = "src";
            version = "1";
            src = pkgs.lib.sources.sourceFilesBySuffices ./. [ ".toml" ];
            taploExtraArgs = "--diff";
          };

          nix-format = pkgs.runCommand "check-format" { } ''
            ${lib.getExe pkgs.nixfmt-rfc-style} --check ${self}
            mkdir -p $out
          '';

          nix-lint = pkgs.runCommand "check-lint" { } ''
            ${lib.getExe pkgs.statix} check
            mkdir -p $out
          '';
        }
        // packages
        // packages_rustfmt
        // packages_clippy;
      }
    );
}
