# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### ✨ Added
- Improved finding ROCm paths, often works without extra configuration now

### ℹ Changed
- `dispatch_ptr` is now available without device-libs

### ❌ Removed
- Removed `amdgpu_device_libs::intrinsics`, use `core::arch::amdgpu` for less common intrinsics instead
- Removed `thin-lto=no` workaround because it is fixed in Rust
- Removed `lto=true` workaround because it is fixed in Rust

## [0.1.0] - 2025-03-25
### ✨ Added
- First release
