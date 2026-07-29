# gpu-kernel Examples

Examples for running Rust code on a GPU.

The most basic example is [`hello_world`](./hello_world) with just a println from the GPU.

[`vector_add`](./vector_add), a classic of GPU examples, does a little more, adding two `Vec`s together on the GPU.

Adding a few performance optimizations to that, like explicitly moving data to and from GPU memory and using larger workgroups is [`vector_add_fast`](./vector_add_fast).

Projects that need more than no-std dependencies can split the code into a GPU/shared `lib.rs` and a `main.rs` as shown in [`split-lib`](./split-lib).
