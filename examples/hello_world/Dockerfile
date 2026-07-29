FROM ubuntu:26.04
# Install ROCm
RUN apt update && apt install -y rocm-dev curl

# Install rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain none -y
ENV PATH="$PATH:/root/.cargo/bin"
# If rustup is already installed, use `rustup component add rust-src`
RUN rustup toolchain install nightly --profile minimal --component rust-src
RUN mkdir /build

# Copy hello world code and libraries
COPY amdgpu-device-libs /build/amdgpu-device-libs
COPY amdgpu-device-libs-build /build/amdgpu-device-libs-build
COPY gpu-kernel /build/gpu-kernel
COPY gpu-kernel-proc-macros /build/gpu-kernel-proc-macros
COPY examples/hello_world /build/examples/hello_world

WORKDIR /build/examples/hello_world
# Build from repo root with: podman build -t gpu-kernel -f examples/hello_world/Dockerfile .
# Then run in: podman run -it --device /dev/kfd --device /dev/dri --net=host --security-opt=no-new-privileges --cap-drop=ALL localhost/gpu-kernel:latest
# CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS="-Ctarget-cpu=gfx<your version>" HIP_PATH=/usr cargo run
