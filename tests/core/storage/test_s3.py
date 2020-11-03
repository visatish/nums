# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pytest
import numpy as np

from nums.core.array.application import ArrayApplication
from nums.core.array.blockarray import BlockArray
from nums.core.storage.storage import StoredArrayS3, ArrayGrid


# Note: We have to skip these tests for now.


@pytest.mark.skip
def test_rwd(app_inst: ArrayApplication):
    array: np.ndarray = np.random.random(35).reshape(7, 5)
    ba: BlockArray = app_inst.array(array, block_shape=(3, 4))
    filename = "darrays/read_write_delete_array_test"
    write_result: BlockArray = app_inst.write_s3(ba, filename)
    write_result_arr = app_inst.get(write_result)
    for grid_entry in write_result.grid.get_entry_iterator():
        assert 'ETag' in write_result_arr[grid_entry]
    ba_read: BlockArray = app_inst.read_s3(filename)
    assert app_inst.get(app_inst.allclose(ba, ba_read))
    delete_result: BlockArray = app_inst.delete_s3(filename)
    delete_result_arr = app_inst.get(delete_result)
    for grid_entry in delete_result.grid.get_entry_iterator():
        deleted_key = delete_result_arr[grid_entry]["Deleted"][0]["Key"]
        assert deleted_key == StoredArrayS3(filename, delete_result.grid).get_key(grid_entry)


@pytest.mark.skip
def test_array_rwd():
    X: np.ndarray = np.random.random(3)
    stored_X = StoredArrayS3("darrays/%s_X" % "__test__")
    stored_X.put_grid(ArrayGrid(shape=X.shape,
                                block_shape=X.shape, dtype=np.float64.__name__))
    stored_X.init_grid()
    stored_X.put_array(X)
    assert np.allclose(X, stored_X.get_array())
    stored_X.del_array()
    stored_X.delete_grid()


def test_grid_copy():
    grid = ArrayGrid(shape=(1, 2),
                     block_shape=(1, 2), dtype=np.float64.__name__)
    assert grid.copy() is not grid


if __name__ == "__main__":
    # pylint: disable=import-error
    from tests import conftest

    app_inst = conftest.get_app("serial")
    test_rwd(app_inst)
    test_array_rwd()
    test_grid_copy()
