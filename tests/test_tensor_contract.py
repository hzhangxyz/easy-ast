#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Hao Zhang<hzhangxyz@outlook.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
from easy_ast.tensor_contract import TensorContract


def test_einsum_1():
    import numpy as np
    b = np.array([1, 2])
    c = np.array([1, 2])
    expect = -np.array([[1, 2], [2, 4]])

    @TensorContract().exec
    def _(i, j):
        a[i, j] = -b[i] * c[j]
        assert np.all(a == expect)


def test_einsum_2():
    import numpy as np
    b = np.array([1, 2])
    c = np.array([1, 2])
    expect = -np.array([[1, 2], [2, 4]])

    @TensorContract(dummy_index={"i", "j"}).exec
    def _():
        d = 1
        assert d == 1
        a[i, j] = -b[i] * c[j]
        assert np.all(a == expect)


def test_einsum_3():
    import numpy as np
    b = np.random.randn(3, 2, 6, 5)
    c = np.random.randn(3, 4)
    d = np.random.randn(5, 4, 2)
    e = np.einsum("ijk,il,klm->mj", b[:, 0], c, d)
    assert e.shape == (2, 6)
    a = np.zeros([6, 2, 2])

    @TensorContract().exec
    def _(i, j, k, l, m):
        x = 1
        # i3, j6, k5, l4, m2
        a[j, m, x] = b[i, 0, j, k] * c[i, l] * d[k, l, m] - e[m, j]
        assert np.sum(np.abs(a)) < 1e-6


def test_einsum_4():
    import numpy as np
    b = np.array([1, 2])
    c = np.array([1, 2])
    d = np.array([[1, 2], [2, 4]]) / 2.

    @TensorContract().exec
    def _(i, j):
        a[i, j] = -b[i] * c[j] + 1 * d[i, j] * (1 + 1)
        assert np.all(a == 0)


def test_einsum_5():
    import numpy as np
    a = np.array([1, 2])
    b = np.array([1, 2])
    x = [0]

    @TensorContract().exec
    def _(i):
        x[0] = a[i] * b[i]
        assert np.all(x[0] == 5)
        y = a[i] * b[i]
        assert np.all(y == 5)
