# Adding numbers on the GPU

Create two `Vec`s with numbers on the CPU and give them to the GPU to add together.
A third `Vec` is passed to write the result.

As passing a mutable slice to a GPU kernel would be illegal in Rust as there must not be multiple mutable references to the same memory, we use a `ThreadIndexedVec` on the GPU side that allows getting a mutable reference to the element for the current thread id but not to other elements of the vector.

The `vector_add_fast` example applies some performance improvements to this example.

See the repo’s readme for how to compile.
