# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from typing import List

import numpy as np

from nums.core.array.blockarray import BlockArray
from nums.core.array import utils as array_utils
from nums.core.storage.storage import ArrayGrid, StoredArray, StoredArrayS3
from nums.core.systems.systems import System
from nums.core.systems.filesystem import FileSystem
from nums.core.array.random import NumsRandomState


class ArrayApplication(object):

    def __init__(self, system: System, filesystem: FileSystem):
        self._system: System = system
        self._filesystem: FileSystem = filesystem
        self._array_grids: (str, ArrayGrid) = {}
        self.one_half = self.scalar(.5)
        self.two = self.scalar(2.0)
        self.one = self.scalar(1.0)
        self.zero = self.scalar(0.0)

    def _get_array_grid(self, filename: str, stored_array_cls) -> ArrayGrid:
        if filename not in self._array_grids:
            store_inst: StoredArray = stored_array_cls(filename)
            self._array_grids[filename] = store_inst.get_grid()
        return self._array_grids[filename]

    ######################################
    # Filesystem API
    ######################################

    def write_fs(self, ba: BlockArray, filename: str):
        res = self._write(ba, filename, self._filesystem.write_block_fs)
        self._filesystem.write_meta_fs(ba, filename)
        return res

    def read_fs(self, filename: str):
        meta = self._filesystem.read_meta_fs(filename)
        addresses = meta["addresses"]
        grid_meta = meta["grid_meta"]
        grid = ArrayGrid.from_meta(grid_meta)
        ba: BlockArray = BlockArray(grid, self._system)
        for grid_entry in addresses:
            node_address = addresses[grid_entry]
            options = {"resources": {node_address: 1.0 / 10 ** 4}}
            ba.blocks[grid_entry].oid = self._filesystem.read_block_fs(filename,
                                                                       grid_entry,
                                                                       grid_meta,
                                                                       options=options)
        return ba

    def delete_fs(self, filename: str):
        meta = self._filesystem.read_meta_fs(filename)
        addresses = meta["addresses"]
        grid_meta = meta["grid_meta"]
        grid = ArrayGrid.from_meta(grid_meta)
        result_grid = ArrayGrid(grid.grid_shape,
                                tuple(np.ones_like(grid.shape, dtype=np.int)),
                                dtype=dict.__name__)
        rarr = BlockArray(result_grid, self._system)
        for grid_entry in addresses:
            node_address = addresses[grid_entry]
            options = {"resources": {node_address: 1.0 / 10 ** 4}}
            rarr.blocks[grid_entry].oid = self._filesystem.delete_block_fs(filename,
                                                                           grid_entry,
                                                                           grid_meta,
                                                                           options=options)
        self._filesystem.delete_meta_fs(filename)
        return rarr

    def write_s3(self, ba: BlockArray, filename: str):
        grid_entry = tuple(np.zeros_like(ba.shape, dtype=np.int))
        result = self._filesystem.write_meta_s3(filename,
                                                grid_meta=ba.grid.to_meta(),
                                                syskwargs={
                                                    "grid_entry": grid_entry,
                                                    "grid_shape": ba.grid.grid_shape
                                                })
        assert "ETag" in self._system.get(result).item(), "Metadata write failed."
        return self._write(ba, filename, self._filesystem.write_block_s3)

    def _write(self, ba: BlockArray, filename, remote_func):
        grid = ba.grid
        result_grid = ArrayGrid(grid.grid_shape,
                                tuple(np.ones_like(grid.shape, dtype=np.int)),
                                dtype=dict.__name__)
        rarr = BlockArray(result_grid, self._system)
        for grid_entry in grid.get_entry_iterator():
            rarr.blocks[grid_entry].oid = remote_func(ba.blocks[grid_entry].oid,
                                                      filename,
                                                      grid_entry,
                                                      grid.to_meta(),
                                                      syskwargs={
                                                          "grid_entry": grid_entry,
                                                          "grid_shape": grid.grid_shape
                                                      })
        return rarr

    def read_s3(self, filename: str):
        store_cls, remote_func = StoredArrayS3, self._filesystem.read_block_s3
        grid = self._get_array_grid(filename, store_cls)
        grid_meta = grid.to_meta()
        grid_entry_iterator = grid.get_entry_iterator()
        rarr = BlockArray(grid, self._system)
        for grid_entry in grid_entry_iterator:
            rarr.blocks[grid_entry].oid = remote_func(filename, grid_entry, grid_meta,
                                                      syskwargs={
                                                          "grid_entry": grid_entry,
                                                          "grid_shape": grid.grid_shape
                                                      })
        return rarr

    def delete_s3(self, filename: str):
        grid = self._get_array_grid(filename, StoredArrayS3)
        grid_entry = tuple(np.zeros_like(grid.shape, dtype=np.int))
        result = self._filesystem.delete_meta_s3(filename,
                                                 syskwargs={
                                                     "grid_entry": grid_entry,
                                                     "grid_shape": grid.grid_shape
                                                 })
        deleted_key = self._system.get(result).item()["Deleted"][0]["Key"]
        assert deleted_key == StoredArrayS3(filename, grid).get_meta_key()
        results: BlockArray = self._delete(filename,
                                           StoredArrayS3,
                                           self._filesystem.delete_block_s3)
        return results

    def _delete(self, filename, store_cls, remote_func):
        grid = self._get_array_grid(filename, store_cls)
        result_grid = ArrayGrid(grid.grid_shape,
                                tuple(np.ones_like(grid.shape, dtype=np.int)),
                                dtype=dict.__name__)
        rarr = BlockArray(result_grid, self._system)
        for grid_entry in grid.get_entry_iterator():
            rarr.blocks[grid_entry].oid = remote_func(filename, grid_entry, grid.to_meta(),
                                                      syskwargs={
                                                          "grid_entry": grid_entry,
                                                          "grid_shape": grid.grid_shape
                                                      })
        return rarr

    def loadtxt(self, fname, dtype=float, comments='# ', delimiter=',',
                converters=None, skiprows=0, usecols=None, unpack=False,
                ndmin=0, encoding='bytes', max_rows=None, num_cpus=4) -> BlockArray:
        return self._filesystem.loadtxt(
            fname, dtype=dtype, comments=comments, delimiter=delimiter,
            converters=converters, skiprows=skiprows,
            usecols=usecols, unpack=unpack, ndmin=ndmin,
            encoding=encoding, max_rows=max_rows, num_cpus=num_cpus)

    ######################################
    # Array Operations API
    ######################################

    def scalar(self, value):
        return BlockArray.from_scalar(value, self._system)

    def array(self, array: np.ndarray, block_shape: tuple = None):
        assert len(array.shape) == len(block_shape)
        return BlockArray.from_np(self._system.compute_imp.from_np(self._system, array),
                                  block_shape=block_shape,
                                  copy=False,
                                  system=self._system)

    def zeros(self, shape: tuple, block_shape: tuple, dtype: np.dtype = None):
        return self._new_array("zeros", shape, block_shape, dtype)

    def ones(self, shape: tuple, block_shape: tuple, dtype: np.dtype = None):
        return self._new_array("ones", shape, block_shape, dtype)

    def empty(self, shape: tuple, block_shape: tuple, dtype: np.dtype = None):
        return self._new_array("empty", shape, block_shape, dtype)

    def _new_array(self, op_name: str, shape: tuple, block_shape: tuple, dtype: np.dtype = None):
        assert len(shape) == len(block_shape)
        if dtype is None:
            dtype = np.float64
        grid = ArrayGrid(shape, block_shape, dtype.__name__)
        grid_meta = grid.to_meta()
        rarr = BlockArray(grid, self._system)
        for grid_entry in grid.get_entry_iterator():
            rarr.blocks[grid_entry].oid = self._system.new_block(op_name,
                                                                 grid_entry,
                                                                 grid_meta,
                                                                 syskwargs={
                                                                     "grid_entry": grid_entry,
                                                                     "grid_shape": grid.grid_shape
                                                                 })
        return rarr

    def concatenate(self, arrays: List, axis: int, axis_block_size: int = None):
        num_arrs = len(arrays)
        assert num_arrs > 1
        first_arr: BlockArray = arrays[0]
        num_axes = len(first_arr.shape)
        # Check assumptions and define result shapes and block shapes.
        for i in range(num_arrs):
            curr_ba: BlockArray = arrays[i]
            assert num_axes == len(curr_ba.shape), "Unequal num axes."
            assert curr_ba.dtype == first_arr.dtype, "Incompatible dtypes " \
                                                     "%s, %s" % (curr_ba.dtype, first_arr.dtype)
            for curr_axis in range(num_axes):
                first_block_size = first_arr.block_shape[curr_axis]
                block_size = curr_ba.block_shape[curr_axis]
                if first_block_size == block_size:
                    continue
                elif axis == curr_axis:
                    assert axis_block_size is not None, "block axis size is required " \
                                                        "when block shapes are neq."
                else:
                    raise ValueError("Other axis shapes and block shapes must be equal.")

        # Compute result shapes.
        result_shape = []
        result_block_shape = []
        for curr_axis in range(num_axes):
            if curr_axis == axis:
                if axis_block_size is None:
                    # They are all equal.
                    axis_block_size = first_arr.block_shape[curr_axis]
                result_block_size = axis_block_size
                result_size = 0
                for i in range(num_arrs):
                    curr_ba: BlockArray = arrays[i]
                    size = curr_ba.shape[curr_axis]
                    result_size += size
            else:
                result_size = first_arr.shape[curr_axis]
                result_block_size = first_arr.block_shape[curr_axis]
            result_shape.append(result_size)
            result_block_shape.append(result_block_size)
        result_shape, result_block_shape = tuple(result_shape), tuple(result_block_shape)
        result_ba = self.empty(result_shape, result_block_shape, first_arr.dtype)

        # Write result blocks.
        # TODO (hme): This can be optimized by updating blocks directly.
        pos = 0
        for arr in arrays:
            delta = arr.shape[axis]
            axis_slice = slice(pos, pos+delta)
            result_selector = tuple([slice(None, None) for _ in range(axis)] + [axis_slice, ...])
            result_ba[result_selector] = arr
            pos += delta
        return result_ba

    def log(self, X: BlockArray):
        return X.ufunc("log")

    def exp(self, X: BlockArray):
        return X.ufunc("exp")

    def abs(self, X: BlockArray):
        return X.ufunc("abs")

    def sum(self, X: BlockArray, axis=0, keepdims=False):
        return X.reduce_axis("sum", axis, keepdims=keepdims)

    def mean(self, X: BlockArray, axis=0, keepdims=False):
        if X.dtype not in (float, np.float32, np.float64):
            X = X.astype(np.float64)
        return self.sum(X, axis=axis, keepdims=keepdims) / X.shape[axis]

    def std(self, X: BlockArray, axis=0, keepdims=False):
        mean = self.mean(X, axis=axis, keepdims=True)
        ss = self.sum((X - mean)**self.two, axis=axis, keepdims=keepdims)
        return self.sqrt(ss / X.shape[axis])

    def sqrt(self, X):
        if X.dtype not in (float, np.float32, np.float64):
            X = X.astype(np.float64)
        return X.ufunc("sqrt")

    def norm(self, X):
        return self.sqrt(X.T @ X)

    def xlogy(self, x: BlockArray, y: BlockArray) -> BlockArray:
        if x.dtype not in (float, np.float32, np.float64):
            x = x.astype(np.float64)
        if x.dtype not in (float, np.float32, np.float64):
            y = y.astype(np.float64)
        return self._block_map_bop("xlogy", x, y)

    def _block_map_bop(self, op_name: str, arr_a: BlockArray, arr_b: BlockArray) -> BlockArray:
        shape = arr_a.shape
        block_shape = arr_a.block_shape
        dtype = array_utils.get_bop_output_type("log", arr_a.dtype, arr_b.dtype)
        assert len(shape) == len(block_shape)
        grid = ArrayGrid(shape, block_shape, dtype.__name__)
        rarr = BlockArray(grid, self._system)
        op = self._system.__getattribute__(op_name)
        for grid_entry in grid.get_entry_iterator():
            rarr.blocks[grid_entry].oid = op(arr_a.blocks[grid_entry].oid,
                                             arr_b.blocks[grid_entry].oid,
                                             syskwargs={
                                                 "grid_entry": grid_entry,
                                                 "grid_shape": grid.grid_shape
                                             })
        return rarr

    def get(self, *arrs):
        if len(arrs) == 1:
            if isinstance(arrs[0], BlockArray):
                return arrs[0].get()
            else:
                return arrs[0]
        else:
            r = []
            for item in arrs:
                if isinstance(item, BlockArray):
                    r.append(item.get())
                else:
                    r.append(item)
            return r

    def allclose(self, a: BlockArray, b: BlockArray, rtol=1.e-5, atol=1.e-8):
        assert a.shape == b.shape and a.block_shape == b.block_shape
        bool_list = []
        grid_shape = a.grid.grid_shape
        for grid_entry in a.grid.get_entry_iterator():
            a_block, b_block = a.blocks[grid_entry].oid, b.blocks[grid_entry].oid
            bool_list.append(self._system.allclose(a_block, b_block, rtol, atol,
                                                   syskwargs={
                                                       "grid_entry": grid_entry,
                                                       "grid_shape": grid_shape
                                                   }))
        oid = self._system.logical_and(*bool_list,
                                       syskwargs={"grid_entry": (0, 0), "grid_shape": (1, 1)})
        return BlockArray.from_oid(oid, (), np.bool, self._system)

    def qr(self, X: BlockArray):
        return self.indirect_tsqr(X)

    def indirect_tsr(self, X: BlockArray, reshape_output=True):
        assert len(X.shape) == 2
        # TODO (hme): This assertion is temporary and ensures returned
        #  shape of qr of block is correct.
        assert X.block_shape[0] >= X.shape[1]
        # Compute R for each block.
        grid = X.grid
        grid_shape = grid.grid_shape
        shape = X.shape
        block_shape = X.block_shape
        R_oids = []
        # Assume no blocking along second dim.
        for i in range(grid_shape[0]):
            # Select a row according to block_shape.
            row = []
            for j in range(grid_shape[1]):
                row.append(X.blocks[i, j].oid)
            R_oids.append(self._system.qr(*row,
                                          mode="r",
                                          axis=1,
                                          syskwargs={
                                              "grid_entry": (i, 0),
                                              "grid_shape": (grid_shape[0], 1),
                                              "options": {"num_return_vals": 1}
                                          })
                          )

        # Construct R by summing over R blocks.
        # TODO (hme): Communication may be inefficient due to redundancy of data.
        R_shape = (shape[1], shape[1])
        R_block_shape = (block_shape[1], block_shape[1])
        tsR = BlockArray(ArrayGrid(shape=R_shape,
                                   block_shape=R_shape,
                                   dtype=X.dtype.__name__),
                         self._system)
        tsR.blocks[0, 0].oid = self._system.qr(*R_oids,
                                               mode="r",
                                               axis=0,
                                               syskwargs={
                                                   "grid_entry": (0, 0),
                                                   "grid_shape": (1, 1),
                                                   "options": {"num_return_vals": 1}
                                               })
        # If blocking is "tall-skinny," then we're done.
        if R_shape != R_block_shape:
            if reshape_output:
                R = tsR.reshape(shape=R_shape, block_shape=R_block_shape)
            else:
                R = tsR
        else:
            R = tsR
        return R

    def indirect_tsqr(self, X: BlockArray, reshape_output=True):
        shape = X.shape
        block_shape = X.block_shape
        R_shape = (shape[1], shape[1])
        R_block_shape = (block_shape[1], block_shape[1])
        tsR = self.indirect_tsr(X, reshape_output=False)

        # Compute inverse of R.
        tsR_inverse = self.inv(tsR)
        # If blocking is "tall-skinny," then we're done.
        if R_shape != R_block_shape:
            R_inverse = tsR_inverse.reshape(shape=R_shape, block_shape=R_block_shape)
            if reshape_output:
                R = tsR.reshape(shape=R_shape, block_shape=R_block_shape)
            else:
                R = tsR
        else:
            R_inverse = tsR_inverse
            R = tsR

        Q = X @ R_inverse
        return Q, R

    def direct_tsqr(self, X, reshape_output=True):
        assert len(X.shape) == 2

        # Compute R for each block.
        shape = X.shape
        grid = X.grid
        grid_shape = grid.grid_shape
        block_shape = X.block_shape
        Q_oids = []
        R_oids = []
        QR_dims = []
        Q2_shape = [0, shape[1]]
        for i in range(grid_shape[0]):
            # Select a row according to block_shape.
            row = []
            for j in range(grid_shape[1]):
                row.append(X.blocks[i, j].oid)
            # We invoke "reduced", so q, r is returned with dimensions (M, K), (K, N), K = min(M, N)
            M = grid.get_block_shape((i, 0))[0]
            N = shape[1]
            K = min(M, N)
            QR_dims.append(((M, K), (K, N)))
            Q2_shape[0] += K
            # Run each row on separate nodes along first axis.
            # This maintains some data locality.
            Q_oid, R_oid = self._system.qr(*row,
                                           mode="reduced",
                                           axis=1,
                                           syskwargs={
                                               "grid_entry": (i, 0),
                                               "grid_shape": (grid_shape[0], 1),
                                               "options": {"num_return_vals": 2}
                                           })
            R_oids.append(R_oid)
            Q_oids.append(Q_oid)

        # TODO (hme): This pulls several order N^2 R matrices on a single node.
        #  A solution is the recursive extension to direct TSQR.
        Q2_oid, R2_oid = self._system.qr(*R_oids,
                                         mode="reduced",
                                         axis=0,
                                         syskwargs={
                                             "grid_entry": (0, 0),
                                             "grid_shape": (1, 1),
                                             "options": {"num_return_vals": 2}
                                         })

        Q2_shape = tuple(Q2_shape)
        Q2_block_shape = (QR_dims[0][1][0], shape[1])
        Q2 = self._vec_from_oids([Q2_oid],
                                 shape=Q2_shape,
                                 block_shape=Q2_block_shape,
                                 dtype=X.dtype)
        # The resulting Q's from this operation are N^2 (same size as above R's).
        Q2_oids = list(map(lambda block: block.oid, Q2.blocks.flatten()))

        # Construct Q.
        Q = self.zeros(shape=shape,
                       block_shape=(block_shape[0], shape[1]),
                       dtype=X.dtype)
        for i, grid_entry in enumerate(Q.grid.get_entry_iterator()):
            Q_dims, R_dims = QR_dims[i]
            Q1_block_shape = Q_dims
            Q2_block_shape = R_dims
            Q.blocks[grid_entry].oid = self._system.bop("tensordot", Q_oids[i], Q2_oids[i],
                                                        a1_shape=Q1_block_shape,
                                                        a2_shape=Q2_block_shape,
                                                        a1_T=False, a2_T=False, axes=1,
                                                        syskwargs={"grid_entry": grid_entry,
                                                                   "grid_shape": Q.grid.grid_shape})

        # Construct R.
        shape = X.shape
        R_shape = (shape[1], shape[1])
        R_block_shape = (block_shape[1], block_shape[1])
        tsR = self._vec_from_oids([R2_oid], shape=R_shape, block_shape=R_shape, dtype=X.dtype)
        # If blocking is "tall-skinny," then we're done.
        if R_shape == R_block_shape or not reshape_output:
            R = tsR
        else:
            R = tsR.reshape(shape=R_shape, block_shape=R_block_shape)

        if Q.shape != block_shape or not reshape_output:
            Q = Q.reshape(shape=shape, block_shape=block_shape)

        return Q, R

    def svd(self, X):
        # TODO(hme): Optimize by merging with direct qr to compute U directly,
        #  to avoid wasting space storing intermediate Q.
        #  This may not really help until we have operator fusion.
        assert len(X.shape) == 2
        block_shape = X.block_shape
        shape = X.shape
        R_shape = (shape[1], shape[1])
        R_block_shape = (block_shape[1], block_shape[1])
        Q, R = self.direct_tsqr(X, reshape_output=False)
        assert R.shape == R.block_shape
        R_U, S, VT = self._system.svd(R.blocks[(0, 0)].oid,
                                      syskwargs={"grid_entry": (0, 0),
                                                 "grid_shape": (1, 1)})
        R_U: BlockArray = self._vec_from_oids([R_U], R_shape, R_block_shape, X.dtype)
        S: BlockArray = self._vec_from_oids([S], R_shape[:1], R_block_shape[:1], X.dtype)
        VT = self._vec_from_oids([VT], R_shape, R_block_shape, X.dtype)
        U = Q @ R_U

        return U, S, VT

    def inverse_triangular(self, X: BlockArray, lower: bool):
        # TODO (hme): Implement scalable version.
        assert X.dtype in (np.float32, np.float64)
        if X.dtype == np.float64:
            lapack_func = self._system.lapack_dtrtri
        elif X.dtype == np.float32:
            lapack_func = self._system.lapack_strtri
        else:
            raise ValueError("Unsupported data type %s" % str(X.dtype))
        return self._inv(lapack_func, {
            "lower": int(lower),
            "unitdiag": 0,
            "overwrite_c": 0
        }, X)

    def inv(self, X: BlockArray):
        return self._inv(self._system.inv, {}, X)

    def _inv(self, remote_func, kwargs, X: BlockArray):
        # TODO (hme): Implement scalable version.
        block_shape = X.block_shape
        assert len(X.shape) == 2
        assert X.shape[0] == X.shape[1]
        single_block = X.shape[0] == X.block_shape[0] and X.shape[1] == X.block_shape[1]
        if single_block:
            result = X.copy()
        else:
            result = X.reshape(block_shape=X.shape)
        result.blocks[0, 0].oid = remote_func(result.blocks[0, 0].oid,
                                              **kwargs,
                                              syskwargs={
                                                  "grid_entry": (0, 0),
                                                  "grid_shape": (1, 1)
                                              })
        if not single_block:
            result = result.reshape(block_shape=block_shape)
        return result

    def cholesky(self, X: BlockArray):
        # TODO (hme): Implement scalable version.
        # Note:
        # A = Q, R
        # A.T @ A = R.T @ R
        # A.T @ A = L @ L.T
        # => R == L.T
        block_shape = X.block_shape
        assert len(X.shape) == 2
        assert X.shape[0] == X.shape[1]
        single_block = X.shape[0] == X.block_shape[0] and X.shape[1] == X.block_shape[1]
        if single_block:
            result = X.copy()
        else:
            result = X.reshape(block_shape=X.shape)
        result.blocks[0, 0].oid = self._system.cholesky(result.blocks[0, 0].oid,
                                                        syskwargs={
                                                            "grid_entry": (0, 0),
                                                            "grid_shape": (1, 1)
                                                        })
        if not single_block:
            result = result.reshape(block_shape=block_shape)
        return result

    def inv_sym_psd(self, X: BlockArray):
        # Assumes X is symmetric PSD.
        # TODO (hme): Implement scalable version.
        assert len(X.shape) == 2
        assert X.shape[0] == X.shape[1]
        single_block = X.shape[0] == X.block_shape[0] and X.shape[1] == X.block_shape[1]
        if single_block:
            result = X.copy()
        else:
            result = X.reshape(block_shape=X.shape)
        result.blocks[0, 0].oid = self._system.inv_sym_psd(result.blocks[0, 0].oid,
                                                           syskwargs={
                                                               "grid_entry": (0, 0),
                                                               "grid_shape": (1, 1)
                                                           })
        if not single_block:
            result = result.reshape(block_shape=X.block_shape)
        return result

    def fast_linear_regression(self, X: BlockArray, y: BlockArray):
        assert len(X.shape) == 2
        assert len(y.shape) == 1
        block_shape = X.block_shape
        shape = X.shape
        R_shape = (shape[1], shape[1])
        R_block_shape = (block_shape[1], block_shape[1])
        Q, R = self.indirect_tsqr(X, reshape_output=False)
        R_inv = self.inv(R)
        if R_shape != R_block_shape:
            R_inv = R_inv.reshape(shape=R_shape, block_shape=R_block_shape)
        theta = R_inv @ (Q.T @ y)
        return theta

    def linear_regression(self, X: BlockArray, y: BlockArray):
        assert len(X.shape) == 2
        assert len(y.shape) == 1
        block_shape = X.block_shape
        shape = X.shape
        R_shape = (shape[1], shape[1])
        R_block_shape = (block_shape[1], block_shape[1])
        Q, R = self.direct_tsqr(X, reshape_output=False)
        # Invert R.
        R_inv = self.inv(R)
        if R_shape != R_block_shape:
            R_inv = R_inv.reshape(shape=R_shape, block_shape=R_block_shape)
        theta = R_inv @ (Q.T @ y)
        return theta

    def ridge_regression(self, X: BlockArray, y: BlockArray, lamb: float):
        assert len(X.shape) == 2
        assert len(y.shape) == 1
        assert lamb >= 0
        block_shape = X.block_shape
        shape = X.shape
        R_shape = (shape[1], shape[1])
        R_block_shape = (block_shape[1], block_shape[1])
        R = self.indirect_tsr(X)
        lamb_vec = self.array(lamb*np.eye(R_shape[0]), block_shape=R_block_shape)
        # TODO (hme): A better solution exists, which inverts R by augmenting X and y.
        #  See Murphy 7.5.2.
        # (lamb_vec + R.T @ R) happens to be symmetric PSD since A.T @ A (= R.T @ R) is PSD
        # and lamb_vec is positive diag.
        theta = self.inv(lamb_vec + R.T @ R) @ (X.T @ y)
        return theta

    def _vec_from_oids(self, oids, shape, block_shape, dtype):
        arr = BlockArray(ArrayGrid(shape=shape,
                                   block_shape=shape,
                                   dtype=dtype.__name__),
                         self._system)
        # Make sure resulting grid shape is a vector (1 dimensional).
        assert np.sum(arr.grid.grid_shape) == (max(arr.grid.grid_shape)
                                               + len(arr.grid.grid_shape) - 1)
        for i, grid_entry in enumerate(arr.grid.get_entry_iterator()):
            arr.blocks[grid_entry].oid = oids[i]
        if block_shape != shape:
            return arr.reshape(block_shape=block_shape)
        return arr

    def random_state(self, seed=None):
        return NumsRandomState(self._system, seed)
