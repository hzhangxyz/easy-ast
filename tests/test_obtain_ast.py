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
import ast
from easy_ast import Statements, Expression, Exec, Eval


def test_get_statement():

    @Statements
    def tree():
        y = x + 1
        z = y * 2

    assert len(tree.body) == 2
    assert isinstance(tree, ast.Module)

    x = 2
    Exec(tree)
    assert locals()["z"] == 6
    # https://docs.python.org/3/library/functions.html#exec
    # Exec will modify locals mapping object, but will not modify the real local variable.


def test_get_expression():

    @Expression
    def tree():
        (x + 1)**2

    assert isinstance(tree, ast.Expression)

    x = 6
    assert Eval(tree) == 49
