#!/usr/bin/env bash
set -euo pipefail

function prepare_ll() {
  # Remove unnecessary stuff
  sd '"target-cpu".* }' ' }' "$1"
  sd '\$(free|malloc) .*' '' "$1"
  sd '@(0|__const.__assert_fail).*' '' "$1"
  sd 'declare .* @llvm.*\n' '' "$1"
  sd '; Function Attrs: .*' '' "$1"
  sd -f ms 'define internal ([^@]* @__ockl_dm_(de)?alloc[^{]*?) \{.*?\n\}' 'declare $1' "$1"
  sd 'define ([^@]* @__amdgpu_util)' 'define linkonce_odr $1' "$1"

  # Remove functions except __util_*
  sd -f ms 'define [^@]* @([^_]|_[^_]|__[^a]|__a[^m]).*?\n\}' '' "$1"
  sd '\n\n+\n' '\n\n' "$1"
  # Check that no other functions are defined
  rg 'define' "$1" | not rg -v __amdgpu_util

  llvm-as "$1"
}


hipcc -c --cuda-device-only --offload-arch=gfx1010 --rocm-path="$ROCM_DEVICE_LIB_PATH" -o util32.bc util.hip -fgpu-rdc -emit-llvm
llvm-dis util32.bc
prepare_ll util32.ll

hipcc -c --cuda-device-only --offload-arch=gfx900 --rocm-path="$ROCM_DEVICE_LIB_PATH" -o util64.bc util.hip -fgpu-rdc -emit-llvm
llvm-dis util64.bc
prepare_ll util64.ll
