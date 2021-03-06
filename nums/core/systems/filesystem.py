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


import os
import pickle
from typing import Any, AnyStr, Tuple, Dict

import numpy as np

from nums.core import settings
from nums.core.storage.storage import ArrayGrid, StoredArrayS3
from nums.core.storage import utils as storage_utils
from nums.core.systems.systems import System
from nums.core.array.blockarray import BlockArray


################
# S3
################
def write_meta_s3(filename: AnyStr, grid_meta: Dict):
    sa: StoredArrayS3 = StoredArrayS3(filename, ArrayGrid.from_meta(grid_meta))
    return np.array(sa.put_grid(sa.grid), dtype=dict)


def delete_meta_s3(filename: AnyStr):
    sa: StoredArrayS3 = StoredArrayS3(filename)
    sa.init_grid()
    return np.array(sa.delete_grid(), dtype=dict)


def write_block_s3(block: Any, filename: AnyStr, grid_entry: Tuple, grid_meta: Dict):
    return np.array(StoredArrayS3(filename, ArrayGrid.from_meta(grid_meta)).put(grid_entry, block),
                    dtype=dict)


def read_block_s3(filename: AnyStr, grid_entry: Tuple, grid_meta: Dict):
    return StoredArrayS3(filename, ArrayGrid.from_meta(grid_meta)).get(grid_entry)


def delete_block_s3(filename: AnyStr, grid_entry: Tuple, grid_meta: Dict):
    return np.array(StoredArrayS3(filename, ArrayGrid.from_meta(grid_meta)).delete(grid_entry),
                    dtype=dict)


##############
# NumPy API
##############
def loadtxt_block(fname, dtype, comments, delimiter,
                  converters, skiprows, usecols, unpack,
                  ndmin, encoding, max_rows):
    return np.loadtxt(
        fname, dtype=dtype, comments=comments, delimiter=delimiter,
        converters=converters, skiprows=skiprows, usecols=usecols, unpack=unpack,
        ndmin=ndmin, encoding=encoding, max_rows=max_rows
    )


###################
# DFS
###################

ARRAY_FILETYPE = "pkl"


def exists(filename: AnyStr):
    return os.path.exists(filename)


def write_meta_fs(meta: Dict, filename: AnyStr):
    """
    Write meta data to disk.
    """
    file_dir = settings.pj(settings.fs_meta, filename)
    settings.Path(file_dir).mkdir(parents=True, exist_ok=True)
    filepath = settings.pj(file_dir, "meta.pkl")
    with open(filepath, "wb") as fh:
        return np.array(pickle.dump(meta, fh), dtype=object)


def read_meta_fs(filename: AnyStr):
    """
    Read meta data from disk.
    """
    file_dir = settings.pj(settings.fs_meta, filename)
    settings.Path(file_dir).mkdir(parents=True, exist_ok=True)
    filepath = settings.pj(file_dir, "meta.pkl")
    with open(filepath, "rb") as fh:
        return pickle.load(fh)


def delete_meta_fs(filename: AnyStr):
    """
    Delete meta data from disk.
    """
    file_dir = settings.pj(settings.fs_meta, filename)
    settings.Path(file_dir).mkdir(parents=True, exist_ok=True)
    filepath = settings.pj(file_dir, "meta.pkl")
    return np.array(os.remove(filepath), dtype=object)


def save(block, filepath):
    if filepath.split(".")[-1] == "npy":
        return np.save(filepath, block)
    elif filepath.split(".")[-1] == "pkl":
        with open(filepath, "wb") as fh:
            return pickle.dump(block, fh)


def load(filepath):
    if filepath.split(".")[-1] == "npy":
        return np.load(filepath)
    elif filepath.split(".")[-1] == "pkl":
        with open(filepath, "rb") as fh:
            return pickle.load(fh)


def write_block_fs(block: Any, filename: AnyStr, grid_entry: Tuple):
    """
    Write block to disk.
    """
    file_dir = settings.pj(settings.fs_data, filename)
    settings.Path(file_dir).mkdir(parents=True, exist_ok=True)
    entry_name = "_".join(list(map(str, grid_entry))) + "." + ARRAY_FILETYPE
    filepath = settings.pj(file_dir, entry_name)
    return np.array(save(block, filepath), dtype=object)


def read_block_fs(filename, grid_entry: Tuple):
    """
    Read block from disk.
    """
    file_dir = settings.pj(settings.fs_data, filename)
    settings.Path(file_dir).mkdir(parents=True, exist_ok=True)
    entry_name = "_".join(list(map(str, grid_entry))) + "." + ARRAY_FILETYPE
    filepath = settings.pj(file_dir, entry_name)
    return load(filepath)


def delete_block_fs(filename, grid_entry: Tuple):
    """
    Delete block from disk.
    """
    file_dir = settings.pj(settings.fs_data, filename)
    settings.Path(file_dir).mkdir(parents=True, exist_ok=True)
    entry_name = "_".join(list(map(str, grid_entry))) + "." + ARRAY_FILETYPE
    filepath = settings.pj(file_dir, entry_name)
    return np.array(os.remove(filepath), dtype=object)


class FileSystem(object):
    # pylint: disable=unused-argument
    # TODO (hme):
    #  - Idempotency for write/delete.
    #  - Write-constraints based on cluster disk capacity.
    #  - Replication of data.
    #  - Less stringent replication of meta-data.
    #  - Journaling?

    def __init__(self, system: System):
        self.system = system
        for func in [write_meta_s3, delete_meta_s3,
                     write_block_s3, read_block_s3, delete_block_s3,
                     write_meta_fs, read_meta_fs, delete_meta_fs,
                     write_block_fs, read_block_fs, delete_block_fs,
                     loadtxt_block]:
            self.system.register(func.__name__, func, {})

    ##################################################
    # Private filesystem-like operations
    ##################################################

    def exists(self, filename: AnyStr, syskwargs):
        raise NotImplementedError()

    ##################################################
    # Block-level (remote) operations
    ##################################################

    def write_meta_s3(self, filename: AnyStr, grid_meta: Dict, syskwargs: Dict):
        return self.system.call("write_meta_s3",
                                filename,
                                grid_meta,
                                syskwargs=syskwargs)

    def delete_meta_s3(self, filename: AnyStr, syskwargs: Dict):
        return self.system.call("delete_meta_s3", filename, syskwargs=syskwargs)

    def write_block_s3(self, block: Any, filename: AnyStr, grid_entry: Tuple, grid_meta: Dict,
                       syskwargs: Dict):
        return self.system.call("write_block_s3",
                                block,
                                filename,
                                grid_entry,
                                grid_meta,
                                syskwargs=syskwargs)

    def read_block_s3(self, filename: AnyStr, grid_entry: Tuple, grid_meta: Dict,
                      syskwargs: Dict):
        return self.system.call("read_block_s3",
                                filename,
                                grid_entry,
                                grid_meta,
                                syskwargs=syskwargs)

    def delete_block_s3(self, filename: AnyStr, grid_entry: Tuple, grid_meta: Dict,
                        syskwargs: Dict):
        return self.system.call("delete_block_s3",
                                filename,
                                grid_entry,
                                grid_meta,
                                syskwargs=syskwargs)

    def loadtxt_block(self, fname, dtype, comments, delimiter,
                      converters, skiprows, usecols, unpack,
                      ndmin, encoding, max_rows, syskwargs: Dict):
        # TODO (hme): Invoke file_exists with options to determine which nodes to pull from.
        return self.system.call("loadtxt_block",
                                fname, dtype, comments, delimiter,
                                converters, skiprows, usecols, unpack,
                                ndmin, encoding, max_rows,
                                syskwargs=syskwargs)

    def write_block_fs(self, block: Any, filename: AnyStr, grid_entry: Tuple, grid_meta: Dict,
                       syskwargs: Dict):
        return self.system.call("write_block_fs",
                                block,
                                filename,
                                grid_entry,
                                syskwargs=syskwargs)

    def read_block_fs(self, filename: AnyStr, grid_entry: Tuple, grid_meta: Dict,
                      options: Dict):
        return self.system.call_with_options("read_block_fs",
                                             args=[filename, grid_entry],
                                             kwargs={},
                                             options=options)

    def delete_block_fs(self, filename: AnyStr, grid_entry: Tuple, grid_meta: Dict,
                        options: Dict):
        return self.system.call_with_options("delete_block_fs",
                                             args=[filename, grid_entry],
                                             kwargs={},
                                             options=options)

    ##################################################
    # Array-level operations
    ##################################################

    def write_meta_fs(self, ba: BlockArray, filename: str):
        addresses: dict = self.system.get_block_addresses(ba.grid)
        meta = {"filename": filename,
                "grid_meta": ba.grid.to_meta(),
                "addresses": addresses}
        oids = []
        for grid_entry in ba.grid.get_entry_iterator():
            node_name = addresses[grid_entry]
            oid = self.system.call_with_options("write_meta_fs",
                                                args=[meta, filename],
                                                kwargs={},
                                                options={"resources": {node_name: 1.0/10**4}})
            oids.append(oid)
        return oids

    def read_meta_fs(self, filename: str):
        for node in self.system.nodes():
            node_key = list(filter(lambda key: "node" in key, node["Resources"].keys()))
            assert len(node_key) == 1
            node_name = node_key[0]
            oid = self.system.call_with_options("read_meta_fs",
                                                args=[filename],
                                                kwargs={},
                                                options={"resources": {node_name: 1.0/10**4}})
            result = self.system.get(oid)
            if result is not None:
                return result
        raise Exception("failed to load metadata.")

    def delete_meta_fs(self, filename: str):
        oids = []
        for node in self.system.nodes():
            node_key = list(filter(lambda key: "node" in key, node["Resources"].keys()))
            assert len(node_key) == 1
            node_name = node_key[0]
            oid = self.system.call_with_options("delete_meta_fs",
                                                args=[filename],
                                                kwargs={},
                                                options={"resources": {node_name: 1.0/10**4}})
            oids.append(oid)
        return oids

    def repartition(self, filename: AnyStr, grid_meta: Dict, syskwargs):
        """
        Repartition a loaded array according to provided grid meta data.
        Implement as simple write then delete sequence.
        In order to delete, need old "grid_meta," which we can read from dfs.
        """
        raise NotImplementedError()

    def loadtxt(self, fname, dtype=float, comments='# ', delimiter=',',
                converters=None, skiprows=0, usecols=None, unpack=False,
                ndmin=0, encoding='bytes', max_rows=None, num_cpus=4) -> BlockArray:
        # pylint: disable=unused-variable
        bytes_per_char, bytes_per_row, bytes_per_col, num_cols = storage_utils.get_np_txt_info(
            fname, comments, delimiter
        )
        chars_per_row = bytes_per_row // bytes_per_char
        assert np.allclose(float(chars_per_row), bytes_per_row / bytes_per_char)
        comment_lines, trailing_newlines = storage_utils.get_np_comments(fname, comments)
        nonrow_chars = trailing_newlines
        for line in comment_lines:
            nonrow_chars += len(line)
        file_size = storage_utils.get_file_size(fname)
        file_chars = file_size // bytes_per_char
        assert np.allclose(float(file_chars), file_size / bytes_per_char)
        row_chars = file_chars - nonrow_chars
        num_rows = row_chars // chars_per_row
        assert np.allclose(float(num_rows), float(row_chars / chars_per_row))
        num_rows_final = num_rows - skiprows
        if max_rows is not None:
            num_rows_final = (num_rows_final, max_rows)
        row_batches: storage_utils.Batch = storage_utils.Batch.from_num_batches(num_rows_final,
                                                                                num_cpus)
        grid = ArrayGrid(shape=(num_rows_final, num_cols),
                         block_shape=(row_batches.batch_size, num_cols),
                         dtype=np.float64.__name__ if dtype is float else dtype.__name__)
        result: BlockArray = BlockArray(grid, system=self.system)
        for i, grid_entry in enumerate(grid.get_entry_iterator()):
            row_start, row_end = row_batches.batches[i]
            batch_skiprows = skiprows + row_start + 1
            batch_max_rows = grid.get_block_shape(grid_entry)[0]
            assert batch_max_rows == row_end - row_start
            result.blocks[grid_entry].oid = self.loadtxt_block(
                fname, dtype=dtype, comments=comments, delimiter=delimiter,
                converters=converters, skiprows=batch_skiprows,
                usecols=usecols, unpack=unpack, ndmin=ndmin,
                encoding=encoding, max_rows=batch_max_rows,
                syskwargs={
                    "grid_entry": grid_entry,
                    "grid_shape": grid.grid_shape
                }
            )
        return result
