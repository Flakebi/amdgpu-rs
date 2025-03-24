//! amdgpu compiler intrinsics.
//!
//! Intrinsics defined for the amdgpu LLVM backend.
//! Availability of intrinsics varies depending on the target architecture.

unsafe extern "C" {
    /// Returns the x coordinate of the workitem index within the workgroup.
    #[link_name = "llvm.amdgcn.workitem.id.x"]
    pub safe fn workitem_id_x() -> u32;
    /// Returns the y coordinate of the workitem index within the workgroup.
    #[link_name = "llvm.amdgcn.workitem.id.y"]
    pub safe fn workitem_id_y() -> u32;
    /// Returns the z coordinate of the workitem index within the workgroup.
    #[link_name = "llvm.amdgcn.workitem.id.z"]
    pub safe fn workitem_id_z() -> u32;

    /// Returns the x coordinate of the workgroup index within the dispatch.
    #[link_name = "llvm.amdgcn.workgroup.id.x"]
    pub safe fn workgroup_id_x() -> u32;
    /// Returns the y coordinate of the workgroup index within the dispatch.
    #[link_name = "llvm.amdgcn.workgroup.id.y"]
    pub safe fn workgroup_id_y() -> u32;
    /// Returns the z coordinate of the workgroup index within the dispatch.
    #[link_name = "llvm.amdgcn.workgroup.id.z"]
    pub safe fn workgroup_id_z() -> u32;

    /// Returns the number of LDS bytes statically allocated for this program.
    #[link_name = "llvm.amdgcn.groupstaticsize"]
    pub safe fn groupstaticsize() -> u32;
    /// Returns the id of the dispatch that is currently executed.
    #[link_name = "llvm.amdgcn.dispatch.id"]
    pub safe fn dispatch_id() -> u64;

    /// Returns the number of threads in a wavefront.
    ///
    /// Is always a power of 2.
    #[link_name = "llvm.amdgcn.wavefrontsize"]
    pub safe fn wavefrontsize() -> u32;

    /// Synchronize all wavefronts in a workgroup.
    ///
    /// Each wavefronts in a workgroup waits at the barrier until all wavefronts in the workgroup arrive at a barrier.
    #[link_name = "llvm.amdgcn.s.barrier"]
    pub safe fn s_barrier();

    /// Sleeps for approximately `count * 64` cycles.
    ///
    /// `count` must be a constant.
    /// Only the lower 7 bits of `count` are used.
    #[link_name = "llvm.amdgcn.s.sleep"]
    pub safe fn s_sleep(count: u32);

    /// Stop execution of the kernel.
    ///
    /// This usually signals an error state.
    #[link_name = "llvm.amdgcn.s.sethalt"]
    pub safe fn s_sethalt(value: u32) -> !;

    /// Masked bit count, low 32 lanes.
    ///
    /// Computes the number of bits set in `value`, masked with a thread mask
    /// which contains 1 for all active threads less than the current thread within a wavefront.
    /// `init` is added to the result.
    #[link_name = "llvm.amdgcn.mbcnt.lo"]
    pub safe fn mbcnt_lo(value: u32, init: u32) -> u32;
    /// Masked bit count, high 32 lanes.
    ///
    /// Computes the number of bits set in `value`, masked with a thread mask
    /// which contains 1 for all active threads less than the current thread within a wavefront.
    /// `init` is added to the result.
    #[link_name = "llvm.amdgcn.mbcnt.hi"]
    pub safe fn mbcnt_hi(value: u32, init: u32) -> u32;

    /// Returns a bitfield (`i32` or `i64`) containing the result of its i1 argument
    /// in all active lanes, and zero in all inactive lanes.
    #[link_name = "llvm.amdgcn.ballot"]
    pub safe fn ballot(b: bool) -> u64;

    /// Indexes into the `value` with the current lane id and returns for each lane
    /// if the corresponding bit is set.
    ///
    /// While [`ballot`] converts a `bool` to a mask, `inverse_ballot` converts a mask back to a `bool`.
    /// This means `inverse_ballot(ballot(b)) == b`.
    /// The inverse of `ballot(inverse_ballot(value)) ~= value` is not always true as inactive lanes are set to zero by `ballot`.
    #[link_name = "llvm.amdgcn.inverse.ballot"]
    pub safe fn inverse_ballot(value: u64) -> bool;

    // The following intrinsics can have multiple sizes

    /// Get `value` from the first active lane in the wavefront.
    #[link_name = "llvm.amdgcn.readfirstlane.i32"]
    pub safe fn readfirstlane_u32(value: u32) -> u32;
    /// Get `value` from the first active lane in the wavefront.
    #[link_name = "llvm.amdgcn.readfirstlane.i64"]
    pub safe fn readfirstlane_u64(value: u64) -> u64;
    /// Get `value` from the lane at index `lane` in the wavefront.
    ///
    /// The lane argument must be uniform across the currently active threads
    /// of the current wavefront. Otherwise, the result is undefined.
    #[link_name = "llvm.amdgcn.readlane.i32"]
    pub fn readlane_u32(value: u32, lane: u32) -> u32;
    /// Get `value` from the lane at index `lane` in the wavefront.
    ///
    /// The lane argument must be uniform across the currently active threads
    /// of the current wavefront. Otherwise, the result is undefined.
    #[link_name = "llvm.amdgcn.readlane.i64"]
    pub fn readlane_u64(value: u64, lane: u64) -> u64;
    /// Return `value` for the lane at index `lane` in the wavefront.
    /// Return `default` for all other lanes.
    ///
    /// The value to write and lane select arguments must be uniform across the
    /// currently active threads of the current wavefront. Otherwise, the result is
    /// undefined.
    ///
    /// `value` is the value returned by `lane`.
    /// `default` is the value returned by all lanes other than `lane`.
    #[link_name = "llvm.amdgcn.writelane.i32"]
    pub fn writelane_u32(value: u32, lane: u32, default: u32) -> u32;
    /// Return `value` for the lane at index `lane` in the wavefront.
    /// Return `default` for all other lanes.
    ///
    /// The value to write and lane select arguments must be uniform across the
    /// currently active threads of the current wavefront. Otherwise, the result is
    /// undefined.
    ///
    /// `value` is the value returned by `lane`.
    /// `default` is the value returned by all lanes other than `lane`.
    #[link_name = "llvm.amdgcn.writelane.i64"]
    pub fn writelane_u64(value: u64, lane: u64, default: u64) -> u64;

    /// Stop execution of the wavefront.
    ///
    /// This usually signals the end of a successful execution.
    #[link_name = "llvm.amdgcn.endpgm"]
    pub safe fn endpgm() -> !;

    /// The `update_dpp` intrinsic represents the `update.dpp` operation in AMDGPU.
    /// It takes an old value, a source operand, a DPP control operand, a row mask, a bank mask, and a bound control.
    /// This operation is equivalent to a sequence of `v_mov_b32` operations.
    ///
    /// `llvm.amdgcn.update.dpp.i32 <old> <src> <dpp_ctrl> <row_mask> <bank_mask> <bound_ctrl>`
    /// Should be equivalent to:
    /// ```asm
    /// v_mov_b32 <dest> <old>
    /// v_mov_b32 <dest> <src> <dpp_ctrl> <row_mask> <bank_mask> <bound_ctrl>
    /// ```
    #[link_name = "llvm.amdgcn.update.dpp.i32"]
    pub fn update_dpp(
        old: u32,
        src: u32,
        dpp_ctrl: u32,
        row_mask: u32,
        bank_mask: u32,
        bound_control: bool,
    ) -> u32;

    /// Measures time based on a fixed frequency.
    ///
    /// Provides a real-time clock counter that runs at constant speed (typically 100 MHz) independent of ALU clock speeds.
    /// The clock is consistent across the chip, so can be used for measuring between different wavefronts.
    #[link_name = "llvm.amdgcn.s.memrealtime"]
    pub safe fn s_memrealtime() -> u64;

    /// Scatter data across all lanes in a wavefront.
    ///
    /// Writes `value` to the lane `lane`.
    ///
    /// Reading from inactive lanes returns `0`.
    /// In case multiple values get written to the same `lane`, the value from the source lane with the higher index is taken.
    #[link_name = "llvm.amdgcn.ds.permute"]
    pub fn ds_permute(lane: u32, value: u32) -> u32;
    /// Gather data across all lanes in a wavefront.
    ///
    /// Returns the `value` given to `ds_permute` by lane `lane`.
    ///
    /// Reading from inactive lanes returns `0`.
    #[link_name = "llvm.amdgcn.ds.bpermute"]
    pub fn ds_bpermute(lane: u32, value: u32) -> u32;
    /// Permute a 64-bit value.
    ///
    /// `selector` selects between different patterns in which the 64-bit value represented by `src0` and `src1` are permuted.
    #[link_name = "llvm.amdgcn.perm"]
    pub fn perm(src0: u32, src1: u32, selector: u32) -> u32;

    // gfx10
    /// Performs arbitrary gather-style operation within a row (16 contiguous lanes) of the second input operand.
    ///
    /// The third and fourth inputs must be uniform across the current wavefront.
    /// These are combined into a single 64-bit value representing lane selects used to swizzle within each row.
    #[link_name = "llvm.amdgcn.permlane16.i32"]
    pub fn permlane16_u32(
        old: u32,
        src0: u32,
        src1: u32,
        src2: u32,
        fi: bool,
        bound_control: bool,
    ) -> u32;

    // gfx10
    /// Performs arbitrary gather-style operation across two rows (16 contiguous lanes) of the second input operand.
    ///
    /// The third and fourth inputs must be uniform across the current wavefront.
    /// These are combined into a single 64-bit value representing lane selects used to swizzle within each row.
    #[link_name = "llvm.amdgcn.permlanex16.i32"]
    pub fn permlanex16_u32(
        old: u32,
        src0: u32,
        src1: u32,
        src2: u32,
        fi: bool,
        bound_control: bool,
    ) -> u32;

    /// Get the index of the current wavefront in the workgroup.
    #[link_name = "llvm.amdgcn.s.get.waveid.in.workgroup"]
    pub safe fn s_get_waveid_in_workgroup() -> u32;

    // gfx10
    /// Clamping atomic subtraction
    ///
    /// Subtract `val` from the value at `addr`, clamping at `0` if the value would become negative.
    /// Returns the value at `addr` before the subtraction.
    #[link_name = "llvm.amdgcn.global.atomic.csub"]
    pub fn global_atomic_csub(addr: *mut u32, val: u32) -> u32;

    // gfx11
    /// Swap `value` between upper and lower 32 lanes in a wavefront.
    ///
    /// Does nothing for wave32.
    #[link_name = "llvm.amdgcn.permlane64"]
    pub fn permlane64_u32(value: u32) -> u32;

    // gfx12
    /// Performs arbitrary gather-style operation within a row (16 contiguous lanes) of the second input operand.
    ///
    /// In contrast to [`permlane16_u32`], allows each lane to specify its own gather lane.
    #[link_name = "llvm.amdgcn.permlane16.var"]
    pub fn permlane16_var(old: u32, src0: u32, src1: u32, fi: bool, bound_control: bool) -> u32;

    // gfx12
    /// Performs arbitrary gather-style operation across two rows (16 contiguous lanes) of the second input operand.
    ///
    /// In contrast to [`permlanex16_u32`], allows each lane to specify its own gather lane.
    #[link_name = "llvm.amdgcn.permlanex16.var"]
    pub fn permlanex16_var(old: u32, src0: u32, src1: u32, fi: bool, bound_control: bool) -> u32;

    // gfx12
    /// Conditional atomic subtraction
    ///
    /// If the value at `addr` is greater or equal than `val`, subtracts `val` from the `value`.
    /// If the value at `addr` is less than `val`, does nothing.
    /// Returns the value at `addr` before the subtraction.
    #[link_name = "llvm.amdgcn.global.atomic.cond.sub"]
    pub fn global_atomic_cond_sub(addr: *mut u32, val: u32) -> u32;

    /// Get the index of the current wavefront in the workgroup.
    #[link_name = "llvm.amdgcn.wave.id"]
    pub safe fn wave_id() -> u32;

    // gfx950
    /// Provide direct access to `v_permlane16_swap_b32` instruction on supported targets.
    ///
    /// Swaps the values across lanes of first 2 operands.
    /// Odd rows of the first operand are swapped with even rows of the second operand (one row is 16 lanes).
    /// Returns a pair for the swapped registers.
    /// The first element of the return corresponds to the swapped element of the first argument.
    #[allow(improper_ctypes)]
    #[link_name = "llvm.amdgcn.permlane16.swap"]
    pub fn permlane16_swap(
        vdst_old: u32,
        vsrc_src0: u32,
        fi: bool,
        bound_control: bool,
    ) -> (u32, u32);

    // gfx950
    /// Provide direct access to `v_permlane32_swap_b32` instruction on supported targets.
    ///
    /// Swaps the values across lanes of first 2 operands.
    /// Rows 2 and 3 of the first operand are swapped with rows 0 and 1 of the second operand (one row is 16 lanes).
    /// Returns a pair for the swapped registers.
    /// The first element of the return corresponds to the swapped element of the first argument.
    #[allow(improper_ctypes)]
    #[link_name = "llvm.amdgcn.permlane32.swap"]
    pub fn permlane32_swap(
        vdst_old: u32,
        vsrc_src0: u32,
        fi: bool,
        bound_control: bool,
    ) -> (u32, u32);
}
