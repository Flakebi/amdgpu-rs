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

        # Common arguments can be set here to avoid repeating them later
        craneArgs =
          isGpu: path:
          let
            # Do not use craneLib.cleanCargoSource, otherwise it does not find util32.bc
            src = if isGpu then ./. else craneLib.cleanCargoSource ./.;
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
            doCheck = !isGpu;

            cargoVendorDir = craneLib.vendorMultipleCargoDeps {
              inherit (craneLib.findCargoFiles src) cargoConfigs;
              cargoLockList = [
                cargoLock

                "${native-toolchain}/lib/rustlib/src/rust/library/Cargo.lock"
              ];
            };

            ROCM_PATH = "${pkgs.rocmPackages.clr}";
            ROCM_DEVICE_LIB_PATH = "${pkgs.rocmPackages.rocm-device-libs}";
            CARGO_BUILD_RUSTFLAGS = "--deny warnings";
          }
          // (
            if !isGpu then
              { }
            else
              {
                CARGO_BUILD_RUSTFLAGS = "--deny warnings -Ctarget-cpu=gfx1036";
              }
          );

        cpu_pkgs = [
          "amdgpu-device-libs-build"
          "examples/default-cpu"
          "examples/hostcall-cpu"
        ];
        gpu_pkgs = [
          "amdgpu-device-libs"
          "examples/hostcall-gpu"
          "examples/panic"
          "examples/println"
          "examples/vector_copy"
        ];

        package_args =
          (lib.genAttrs cpu_pkgs (pkg: craneArgs false pkg))
          // (lib.genAttrs gpu_pkgs (pkg: craneArgs true pkg));

        # Build the actual crate itself, reusing the dependency artifacts.
        packages = builtins.mapAttrs (
          pkg: args: craneLib.buildPackage (args // { cargoArtifacts = craneLib.buildDepsOnly args; })
        ) package_args;

        packages_rustfmt = builtins.listToAttrs (
          builtins.map (pkg: {
            name = "${pkg}-fmt";
            value = craneLib.cargoFmt { src = craneLib.cleanCargoSource ./${pkg}; };
          }) (cpu_pkgs ++ gpu_pkgs)
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
            default-cpu = lib.getExe' packages."examples/default-cpu" "default-cpu";
            hostcall-cpu = lib.getExe' packages."examples/hostcall-cpu" "hostcall-cpu";
          in
          pkgs.writeShellScriptBin "runExamples" ''
            set -euxo pipefail
            ${default-cpu} ${packages."examples/vector_copy"}/lib/vector_copy.elf "$@"
            ${default-cpu} ${packages."examples/println"}/lib/println.elf "$@"
            ${hostcall-cpu} ${packages."examples/hostcall-gpu"}/lib/hostcall_gpu.elf "$@"
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
