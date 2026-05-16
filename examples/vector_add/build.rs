fn main() {
    #[cfg(feature = "gpu")]
    amdgpu_device_libs_build::build();
}
