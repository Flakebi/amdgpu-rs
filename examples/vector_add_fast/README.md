# Adding numbers on the GPU more quickly

Based on the `vector_add` example with some performance improvements.
To show measurable impact of these, the added vectors are a lot larger and the sum formula is more complicated.

The two source `Vec`s are initialized on the CPU and then copied to GPU memory.
The third, result `Vec` is left uninitialized, as all elements are written by the GPU kernel, and passed as a raw pointer to the GPU.

See the repo’s readme for how to compile.
