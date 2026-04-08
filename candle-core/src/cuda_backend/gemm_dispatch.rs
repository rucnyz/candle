//! Pluggable GPU GEMM dispatch — replaces cuBLAS when the `cublas` feature is disabled.
//!
//! Users register a GEMM function at startup via `register_gemm_dispatch()`.
//! All `Tensor::matmul()` on CUDA then routes through the registered function.

use std::ffi::c_void;
use std::sync::OnceLock;

/// Signature for a strided batched GEMM dispatch function.
///
/// Computes D = A @ B using cuBLAS-style column-major conventions.
///
/// Parameters follow cuBLAS `gemm_strided_batched` conventions:
/// - `a`, `b`, `d`: device pointers (already offset to start)
/// - `m`, `n`, `k`: problem dimensions
/// - `batch`: number of batch elements (1 for non-batched)
/// - `lda`, `ldb`, `ldd`: leading dimensions
/// - `stride_a`, `stride_b`, `stride_d`: batch strides
/// - `transa`, `transb`: whether A/B are transposed
/// - `dtype`: 0 = BF16, 1 = F16, 2 = F32, 3 = F64
/// - `stream`: CUDA stream pointer
///
/// Returns 0 on success, nonzero on error.
pub type GemmDispatchFn = unsafe fn(
    a: *const c_void,
    b: *const c_void,
    d: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    batch: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    stride_a: i64,
    stride_b: i64,
    stride_d: i64,
    transa: bool,
    transb: bool,
    dtype: u32,
    stream: *const c_void,
) -> i32;

static GEMM_DISPATCH: OnceLock<GemmDispatchFn> = OnceLock::new();

/// Register a custom GEMM dispatch function.
///
/// Must be called before any GPU matmul when the `cublas` feature is disabled.
/// When `cublas` is enabled, this is ignored and cuBLAS is used directly.
pub fn register_gemm_dispatch(f: GemmDispatchFn) {
    GEMM_DISPATCH.set(f).ok();
}

/// Call the registered GEMM dispatch.
pub(crate) fn call_gemm(
    a: *const c_void,
    b: *const c_void,
    d: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    batch: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    stride_a: i64,
    stride_b: i64,
    stride_d: i64,
    transa: bool,
    transb: bool,
    dtype: u32,
    stream: *const c_void,
) -> std::result::Result<(), String> {
    let f = GEMM_DISPATCH.get().ok_or_else(|| {
        "No GPU GEMM backend registered. Call register_gemm_dispatch() at startup, \
         or enable the `cublas` feature."
            .to_string()
    })?;
    let ret = unsafe {
        f(
            a, b, d, m, n, k, batch, lda, ldb, ldd, stride_a, stride_b, stride_d, transa, transb,
            dtype, stream,
        )
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(format!(
            "GEMM dispatch failed (code {ret}) m={m} n={n} k={k} batch={batch} dtype={dtype}"
        ))
    }
}
