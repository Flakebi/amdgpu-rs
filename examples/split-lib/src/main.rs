use gpu_kernel::LaunchConfig;

use split_lib::kernel;

fn main() {
    // Only heap variables are shared on dedicated AMD GPUs,
    // so create the string on the heap
    let s = "World".to_string();

    kernel.launch(
        LaunchConfig::new()
            .threads_per_workgroup([10, 1, 1])
            .workgroups([1, 1, 1]),
        &s,
    );
}
