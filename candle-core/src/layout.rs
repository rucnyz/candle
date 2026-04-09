//! Tensor Layouts including contiguous or sparse strides
use crate::{Error, Result, Shape};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Layout {
    shape: Shape,
    // The strides are given in number of elements and not in bytes.
    stride: Vec<usize>,
    start_offset: usize,
}

impl Layout {
    pub fn new(shape: Shape, stride: Vec<usize>, start_offset: usize) -> Self {
        Self {
            shape,
            stride,
            start_offset,
        }
    }

    pub fn contiguous_with_offset<S: Into<Shape>>(shape: S, start_offset: usize) -> Self {
        let shape = shape.into();
        let stride = shape.stride_contiguous();
        Self {
            shape,
            stride,
            start_offset,
        }
    }

    pub fn contiguous<S: Into<Shape>>(shape: S) -> Self {
        Self::contiguous_with_offset(shape, 0)
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// The dimension size for a specified dimension index.
    pub fn dim<D: crate::shape::Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(&self.shape, "dim")?;
        Ok(self.dims()[dim])
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    /// Returns the appropriate start and stop offset if the data is stored in a C
    /// contiguous (aka row major) way.
    pub fn contiguous_offsets(&self) -> Option<(usize, usize)> {
        if self.is_contiguous() {
            let start_o = self.start_offset;
            Some((start_o, start_o + self.shape.elem_count()))
        } else {
            None
        }
    }

    /// Returns true if the data is stored in a C contiguous (aka row major) way.
    /// Note that this does not implies that the start offset is 0 or that there are no extra
    /// elements at the end of the storage.
    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.stride)
    }

    /// Returns true if the data is stored in a Fortran contiguous (aka column major) way.
    pub fn is_fortran_contiguous(&self) -> bool {
        self.shape.is_fortran_contiguous(&self.stride)
    }

    /// Compute strides for reshaping without a data copy (PyTorch's `view` semantics).
    ///
    /// Returns `Some(new_strides)` when the current (possibly non-contiguous) layout
    /// can be reinterpreted as `new_shape` by just changing metadata.
    /// Returns `None` when an actual data copy is required.
    ///
    /// Algorithm: partition old dims into maximal contiguous chunks (right-to-left),
    /// then check that `new_shape` re-factors those same chunks.
    /// Port of PyTorch `computeStride_impl` from `aten/src/ATen/TensorUtils.cpp`.
    pub fn strided_reshape(&self, new_shape: &[usize]) -> Option<Vec<usize>> {
        let old_shape = self.shape.dims();
        let old_stride = &self.stride;
        if old_shape.is_empty() {
            return Some(vec![1; new_shape.len()]);
        }

        let mut new_stride = vec![0usize; new_shape.len()];
        let mut view_d = new_shape.len() as isize - 1;

        // Base stride of the current contiguous chunk (starts at innermost).
        let mut chunk_base_stride = *old_stride.last().unwrap();
        let mut tensor_numel: usize = 1;
        let mut view_numel: usize = 1;

        for tensor_d in (0..old_shape.len()).rev() {
            tensor_numel *= old_shape[tensor_d];

            // Detect chunk boundary: either leftmost dim, or next dim breaks contiguity.
            let is_boundary = tensor_d == 0
                || (old_shape[tensor_d - 1] != 1
                    && old_stride[tensor_d - 1] != tensor_numel * chunk_base_stride);

            if is_boundary {
                // Consume new dims right-to-left until view_numel matches tensor_numel.
                while view_d >= 0
                    && (view_numel < tensor_numel || new_shape[view_d as usize] == 1)
                {
                    let vd = view_d as usize;
                    new_stride[vd] = view_numel * chunk_base_stride;
                    view_numel *= new_shape[vd];
                    view_d -= 1;
                }

                if view_numel != tensor_numel {
                    return None;
                }

                // Reset for next chunk.
                if tensor_d > 0 {
                    chunk_base_stride = old_stride[tensor_d - 1];
                    tensor_numel = 1;
                    view_numel = 1;
                }
            }
        }

        if view_d != -1 {
            return None;
        }
        Some(new_stride)
    }

    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let dims = self.shape().dims();
        if dim >= dims.len() {
            Err(Error::DimOutOfRange {
                shape: self.shape().clone(),
                dim: dim as i32,
                op: "narrow",
            }
            .bt())?
        }
        if start + len > dims[dim] {
            Err(Error::NarrowInvalidArgs {
                shape: self.shape.clone(),
                dim,
                start,
                len,
                msg: "start + len > dim_len",
            }
            .bt())?
        }
        let mut dims = dims.to_vec();
        dims[dim] = len;
        Ok(Self {
            shape: Shape::from(dims),
            stride: self.stride.clone(),
            start_offset: self.start_offset + self.stride[dim] * start,
        })
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self> {
        let rank = self.shape.rank();
        if rank <= dim1 || rank <= dim2 {
            Err(Error::UnexpectedNumberOfDims {
                expected: usize::max(dim1, dim2),
                got: rank,
                shape: self.shape().clone(),
            }
            .bt())?
        }
        let mut stride = self.stride().to_vec();
        let mut dims = self.shape().dims().to_vec();
        dims.swap(dim1, dim2);
        stride.swap(dim1, dim2);
        Ok(Self {
            shape: Shape::from(dims),
            stride,
            start_offset: self.start_offset,
        })
    }

    pub fn permute(&self, idxs: &[usize]) -> Result<Self> {
        let is_permutation =
            idxs.len() == self.shape.rank() && (0..idxs.len()).all(|i| idxs.contains(&i));
        if !is_permutation {
            crate::bail!(
                "dimension mismatch in permute, tensor {:?}, dims: {:?}",
                self.dims(),
                idxs
            )
        }
        let stride = self.stride();
        let dims = self.shape().dims();
        let mut perm_stride = stride.to_vec();
        let mut perm_dims = dims.to_vec();
        for (i, &idx) in idxs.iter().enumerate() {
            perm_stride[i] = stride[idx];
            perm_dims[i] = dims[idx];
        }
        Ok(Self {
            shape: Shape::from(perm_dims),
            stride: perm_stride,
            start_offset: self.start_offset,
        })
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let shape = shape.into();
        if shape.rank() < self.shape().rank() {
            return Err(Error::BroadcastIncompatibleShapes {
                src_shape: self.shape().clone(),
                dst_shape: shape,
            }
            .bt());
        }
        let added_dims = shape.rank() - self.shape().rank();
        let mut stride = vec![0; added_dims];
        for (&dst_dim, (&src_dim, &src_stride)) in shape.dims()[added_dims..]
            .iter()
            .zip(self.dims().iter().zip(self.stride()))
        {
            let s = if dst_dim == src_dim {
                src_stride
            } else if src_dim != 1 {
                return Err(Error::BroadcastIncompatibleShapes {
                    src_shape: self.shape().clone(),
                    dst_shape: shape,
                }
                .bt());
            } else {
                0
            };
            stride.push(s)
        }
        Ok(Self {
            shape,
            stride,
            start_offset: self.start_offset,
        })
    }

    pub(crate) fn strided_index(&self) -> crate::StridedIndex<'_> {
        crate::StridedIndex::from_layout(self)
    }

    pub(crate) fn strided_blocks(&self) -> crate::StridedBlocks<'_> {
        let mut block_len = 1;
        let mut contiguous_dims = 0; // These are counted from the right.
        for (&stride, &dim) in self.stride().iter().zip(self.dims().iter()).rev() {
            if stride != block_len {
                break;
            }
            block_len *= dim;
            contiguous_dims += 1;
        }
        let index_dims = self.dims().len() - contiguous_dims;
        if index_dims == 0 {
            crate::StridedBlocks::SingleBlock {
                start_offset: self.start_offset,
                len: block_len,
            }
        } else {
            let block_start_index = crate::StridedIndex::new(
                &self.dims()[..index_dims],
                &self.stride[..index_dims],
                self.start_offset,
            );
            crate::StridedBlocks::MultipleBlocks {
                block_start_index,
                block_len,
            }
        }
    }

    // Returns the contiguous offsets with broadcast if applicable.
    pub(crate) fn offsets_b(&self) -> Option<ContiguousOffsetsWithBroadcast> {
        let mut left_broadcast = 1;
        let mut right_broadcast = 1;
        let strides = self.stride();
        let dims = self.dims();
        let mut start_cont = 0;
        let mut end_cont = dims.len();
        for (&s, &d) in strides.iter().zip(dims.iter()) {
            if s != 0 {
                break;
            }
            start_cont += 1;
            left_broadcast *= d;
        }
        if start_cont == dims.len() {
            return Some(ContiguousOffsetsWithBroadcast {
                start: self.start_offset,
                len: 1,
                left_broadcast,
                right_broadcast: 1,
            });
        }
        for (&s, &d) in strides.iter().zip(dims.iter()).rev() {
            if s != 0 {
                break;
            }
            end_cont -= 1;
            right_broadcast *= d;
        }
        // Check that the inner dims are contiguous
        let strides = &strides[start_cont..end_cont];
        let dims = &dims[start_cont..end_cont];
        let mut len = 1;
        for (&stride, &dim) in strides.iter().zip(dims.iter()).rev() {
            if stride != len {
                return None;
            }
            len *= dim;
        }
        Some(ContiguousOffsetsWithBroadcast {
            start: self.start_offset,
            len,
            left_broadcast,
            right_broadcast,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(shape: &[usize], stride: &[usize], offset: usize) -> Layout {
        Layout::new(Shape::from_dims(shape), stride.to_vec(), offset)
    }

    #[test]
    fn strided_reshape_contiguous() {
        // [4, 6144] contiguous → [4, 48, 128] should work (4*48*128 = 4*6144 = 24576)
        let l = layout(&[4, 6144], &[6144, 1], 0);
        let s = l.strided_reshape(&[4, 48, 128]).unwrap();
        assert_eq!(s, vec![6144, 128, 1]);
    }

    #[test]
    fn strided_reshape_narrow_then_split() {
        // Simulate: [4, 6144].narrow(1, 0, 4096) → [4, 4096] stride [6144, 1]
        // Then reshape to [4, 32, 128]
        let l = layout(&[4, 4096], &[6144, 1], 0);
        let s = l.strided_reshape(&[4, 32, 128]).unwrap();
        // dim2: stride=1, dim1: stride=128, dim0: stride=6144 (from original)
        assert_eq!(s, vec![6144, 128, 1]);
    }

    #[test]
    fn strided_reshape_narrow_kv() {
        // K after narrow: [4, 1024] stride [6144, 1], offset=4096
        let l = layout(&[4, 1024], &[6144, 1], 4096);
        let s = l.strided_reshape(&[4, 8, 128]).unwrap();
        assert_eq!(s, vec![6144, 128, 1]);
    }

    #[test]
    fn strided_reshape_incompatible() {
        // [4, 4096] stride [6144, 1] → reshape to [16384] requires contiguous
        let l = layout(&[4, 4096], &[6144, 1], 0);
        assert!(l.strided_reshape(&[16384]).is_none());
    }

    #[test]
    fn strided_reshape_transpose_incompatible() {
        // Transposed [4, 3] stride [1, 4] → reshape to [12] should fail
        let l = layout(&[4, 3], &[1, 4], 0);
        assert!(l.strided_reshape(&[12]).is_none());
    }

    #[test]
    fn strided_reshape_size_one_dims() {
        // [1, 4096] stride [4096, 1] → [4096] should work
        let l = layout(&[1, 4096], &[4096, 1], 0);
        let s = l.strided_reshape(&[4096]).unwrap();
        assert_eq!(s, vec![1]);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContiguousOffsetsWithBroadcast {
    pub start: usize,
    pub len: usize,
    pub left_broadcast: usize,
    pub right_broadcast: usize,
}
