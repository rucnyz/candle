//! TensorHook — external override point for all tensor operations.
//!
//! Register a hook at startup; candle checks it before its own dispatch.
//! If the hook returns `Some(result)`, candle uses it. `None` → default path.
//!
//! ```ignore
//! struct MyOps;
//! impl TensorHook for MyOps {
//!     fn exp(&self, x: &Tensor) -> Option<Result<Tensor>> { /* custom kernel */ }
//! }
//! static MY_OPS: MyOps = MyOps;
//! candle_core::hook::set(Some(&MY_OPS));
//! ```

use crate::{DType, Result, Tensor};
use std::cell::Cell;

/// Trait for overriding tensor operations externally.
///
/// All methods default to `None` (= fall through to candle's implementation).
/// Override only the ops you care about.
#[allow(unused_variables)]
pub trait TensorHook: Send + Sync {
    // ── Unary ops ────────────────────────────────────────────────
    fn exp(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn log(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn sin(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn cos(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn tanh(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn abs(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn neg(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn recip(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn sqr(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn sqrt(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn gelu(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn gelu_erf(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn erf(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn relu(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn silu(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn ceil(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn floor(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn round(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn sign(&self, x: &Tensor) -> Option<Result<Tensor>> { None }

    // ── Binary ops ───────────────────────────────────────────────
    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Option<Result<Tensor>> { None }
    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Option<Result<Tensor>> { None }
    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Option<Result<Tensor>> { None }
    fn div(&self, lhs: &Tensor, rhs: &Tensor) -> Option<Result<Tensor>> { None }
    fn maximum(&self, lhs: &Tensor, rhs: &Tensor) -> Option<Result<Tensor>> { None }
    fn minimum(&self, lhs: &Tensor, rhs: &Tensor) -> Option<Result<Tensor>> { None }

    // ── Core compute ops ─────────────────────────────────────────
    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Option<Result<Tensor>> { None }
    fn contiguous(&self, x: &Tensor) -> Option<Result<Tensor>> { None }
    fn to_dtype(&self, x: &Tensor, dtype: DType) -> Option<Result<Tensor>> { None }
    fn copy_strided(&self, x: &Tensor) -> Option<Result<Tensor>> { None }

    // ── Indexing ops ─────────────────────────────────────────────
    fn index_select(&self, x: &Tensor, ids: &Tensor, dim: usize) -> Option<Result<Tensor>> { None }
    fn gather(&self, x: &Tensor, ids: &Tensor, dim: usize) -> Option<Result<Tensor>> { None }

    // ── Reduction ops ────────────────────────────────────────────
    fn sum(&self, x: &Tensor, dims: &[usize]) -> Option<Result<Tensor>> { None }
    fn max(&self, x: &Tensor, dims: &[usize]) -> Option<Result<Tensor>> { None }
    fn min(&self, x: &Tensor, dims: &[usize]) -> Option<Result<Tensor>> { None }
    fn argmax(&self, x: &Tensor, dim: usize) -> Option<Result<Tensor>> { None }
    fn argmin(&self, x: &Tensor, dim: usize) -> Option<Result<Tensor>> { None }

    // ── Compose ops ──────────────────────────────────────────────
    fn cat(&self, tensors: &[&Tensor], dim: usize) -> Option<Result<Tensor>> { None }
    fn where_cond(&self, cond: &Tensor, on_true: &Tensor, on_false: &Tensor) -> Option<Result<Tensor>> { None }

    // ── Misc ─────────────────────────────────────────────────────
    fn affine(&self, x: &Tensor, mul: f64, add: f64) -> Option<Result<Tensor>> { None }
    fn powf(&self, x: &Tensor, e: f64) -> Option<Result<Tensor>> { None }
    fn elu(&self, x: &Tensor, alpha: f64) -> Option<Result<Tensor>> { None }
    fn to_device(&self, x: &Tensor, device: &crate::Device) -> Option<Result<Tensor>> { None }
}

// ── Thread-local hook storage ────────────────────────────────────

thread_local! {
    static HOOK: Cell<Option<&'static dyn TensorHook>> = const { Cell::new(None) };
}

/// Register (or clear) the global tensor hook for the current thread.
pub fn set(hook: Option<&'static dyn TensorHook>) {
    HOOK.with(|c| c.set(hook));
}

/// Get the currently registered hook (if any).
pub fn get() -> Option<&'static dyn TensorHook> {
    HOOK.with(|c| c.get())
}

// ── Dispatch helper with recursion guard ─────────────────────────

/// Try dispatching through the hook. Clears the hook during the call
/// to prevent infinite recursion if the hook itself uses candle ops.
///
/// Usage in tensor methods:
/// ```ignore
/// if let Some(r) = crate::hook::dispatch(|h| h.exp(self)) {
///     return r;
/// }
/// ```
#[inline]
pub fn dispatch<F>(f: F) -> Option<Result<Tensor>>
where
    F: FnOnce(&dyn TensorHook) -> Option<Result<Tensor>>,
{
    HOOK.with(|cell| {
        let hook = cell.get()?;
        cell.set(None); // recursion guard
        let result = f(hook);
        cell.set(Some(hook)); // restore
        result
    })
}
