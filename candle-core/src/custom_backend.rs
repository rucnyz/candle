//! Custom backend — stub implementation for externally-provided backends.
//!
//! All compute is handled via TensorHook; this backend only needs to hold data.
//! Every BackendStorage/BackendDevice method that would compute bails — the hook
//! layer intercepts at the Tensor level before we ever reach here.
#![allow(dead_code)]

use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use std::any::Any;
use std::fmt;
use std::sync::Arc;

// ── CustomDevice ────────────────────────────────────────────────

/// Opaque device handle for externally-provided backends.
///
/// The actual device state (e.g. KfdDevice, HipDevice) is stored inside
/// `inner` and downcasted by the external crate.
#[derive(Clone)]
pub struct CustomDevice {
    /// Opaque device state, type-erased.
    inner: Arc<dyn Any + Send + Sync>,
    /// Device ordinal (for DeviceLocation).
    ordinal: usize,
    /// Human-readable name (e.g. "amd:0").
    name: String,
}

impl fmt::Debug for CustomDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CustomDevice({})", self.name)
    }
}

impl CustomDevice {
    /// Create a new custom device with an opaque inner state.
    pub fn new<T: Any + Send + Sync>(inner: T, ordinal: usize, name: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(inner),
            ordinal,
            name: name.into(),
        }
    }

    /// Downcast the inner state to a concrete type.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.inner.downcast_ref()
    }

    /// Get the device ordinal.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}

// ── CustomStorage ───────────────────────────────────────────────

/// Opaque storage for externally-provided backends.
///
/// The actual GPU buffer is stored inside `inner` and downcasted by the
/// external crate via the TensorHook.
pub struct CustomStorage {
    /// Opaque storage state (e.g. GpuBuffer, HipBuffer).
    inner: Arc<dyn Any + Send + Sync>,
    dtype: DType,
    device: CustomDevice,
}

impl fmt::Debug for CustomStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CustomStorage({:?}, {:?})", self.dtype, self.device)
    }
}

impl CustomStorage {
    /// Create a new custom storage with an opaque inner state.
    pub fn new<T: Any + Send + Sync>(inner: T, dtype: DType, device: CustomDevice) -> Self {
        Self {
            inner: Arc::new(inner),
            dtype,
            device,
        }
    }

    /// Downcast the inner state to a concrete type.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.inner.downcast_ref()
    }

    /// Get a clone of the inner Arc (for sharing storage between tensors).
    pub fn inner_arc(&self) -> &Arc<dyn Any + Send + Sync> {
        &self.inner
    }
}

// ── BackendStorage impl ─────────────────────────────────────────

macro_rules! bail_custom {
    ($op:expr) => {
        crate::bail!(concat!("custom backend: ", $op, " — use TensorHook to override"))
    };
}

impl crate::backend::BackendStorage for CustomStorage {
    type Device = CustomDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        // Allow cloning — just Arc-clone the inner.
        Ok(Self {
            inner: self.inner.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
        })
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        bail_custom!("to_cpu_storage")
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        bail_custom!("const_set")
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        bail_custom!("affine")
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        bail_custom!("powf")
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        bail_custom!("elu")
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        bail_custom!("reduce_op")
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        bail_custom!("cmp")
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        bail_custom!("to_dtype")
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        bail_custom!("unary_impl")
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        bail_custom!("binary_impl")
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        bail_custom!("where_cond")
    }

    fn conv1d(
        &self, _: &Layout, _: &Self, _: &Layout, _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        bail_custom!("conv1d")
    }

    fn conv_transpose1d(
        &self, _: &Layout, _: &Self, _: &Layout, _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        bail_custom!("conv_transpose1d")
    }

    fn conv2d(
        &self, _: &Layout, _: &Self, _: &Layout, _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        bail_custom!("conv2d")
    }

    fn conv_transpose2d(
        &self, _: &Layout, _: &Self, _: &Layout, _: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        bail_custom!("conv_transpose2d")
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        bail_custom!("index_select")
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        bail_custom!("gather")
    }

    fn scatter_set(
        &mut self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize,
    ) -> Result<()> {
        bail_custom!("scatter_set")
    }

    fn scatter_add_set(
        &mut self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize,
    ) -> Result<()> {
        bail_custom!("scatter_add_set")
    }

    fn index_add(
        &self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout, _: usize,
    ) -> Result<Self> {
        bail_custom!("index_add")
    }

    fn matmul(
        &self, _: &Self, _: (usize, usize, usize, usize), _: &Layout, _: &Layout,
    ) -> Result<Self> {
        bail_custom!("matmul")
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        bail_custom!("copy_strided_src")
    }

    fn copy2d(
        &self, _: &mut Self, _: usize, _: usize, _: usize, _: usize, _: usize, _: usize,
    ) -> Result<()> {
        bail_custom!("copy2d")
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        bail_custom!("avg_pool2d")
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        bail_custom!("max_pool2d")
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        bail_custom!("upsample_nearest1d")
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        bail_custom!("upsample_nearest2d")
    }

    fn upsample_bilinear2d(
        &self, _: &Layout, _: usize, _: usize, _: bool, _: Option<f64>, _: Option<f64>,
    ) -> Result<Self> {
        bail_custom!("upsample_bilinear2d")
    }
}

// ── BackendDevice impl ──────────────────────────────────────────

impl crate::backend::BackendDevice for CustomDevice {
    type Storage = CustomStorage;

    fn new(_ordinal: usize) -> Result<Self> {
        // Can't create a custom device without inner state.
        // Use CustomDevice::new() directly instead.
        crate::bail!("CustomDevice::new() requires explicit inner state, use CustomDevice::new(inner, ordinal, name)")
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Custom {
            gpu_id: self.ordinal,
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal && self.name == other.name
    }

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        bail_custom!("zeros_impl")
    }

    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        bail_custom!("alloc_uninit")
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        bail_custom!("storage_from_slice")
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        bail_custom!("storage_from_cpu_storage")
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        bail_custom!("storage_from_cpu_storage_owned")
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        bail_custom!("rand_uniform")
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        bail_custom!("rand_normal")
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        bail_custom!("set_seed")
    }

    fn get_current_seed(&self) -> Result<u64> {
        bail_custom!("get_current_seed")
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
