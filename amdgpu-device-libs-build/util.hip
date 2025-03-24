// ./compile-util.sh

#include <hip/hip_runtime.h>

extern "C" __device__ void __amdgpu_util_dealloc(void *addr) {
    free(addr);
}

extern "C" __device__ void *__amdgpu_util_alloc(ulong sz) {
    return (void *)malloc(sz);
}

extern "C" __device__ void __amdgpu_util_print_stdout(const char *str, int sz) {
    printf("%.*s", sz, str);
}

extern "C" __device__ void *__amdgpu_util_dispatch_ptr() {
    return (void *)__builtin_amdgcn_dispatch_ptr();
}

extern "C" __device__ void *__amdgpu_util_queue_ptr() {
    return (void *)__builtin_amdgcn_queue_ptr();
}

extern "C" __device__ void *__amdgpu_util_kernarg_segment_ptr() {
    return (void *)__builtin_amdgcn_kernarg_segment_ptr();
}

extern "C" __device__ void *__amdgpu_util_implicitarg_ptr() {
    return (void *)__builtin_amdgcn_implicitarg_ptr();
}
