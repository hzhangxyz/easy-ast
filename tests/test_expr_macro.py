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
from easy_ast import *


class MultRightPlus(Macro):

    def __init__(self, plus):
        super().__init__()
        self.plus = plus

    def visit_BinOp(self, node):
        match node.op:
            case ast.Mult():
                result = ast.BinOp(
                    op=node.op,
                    left=node.left,
                    right=ast.BinOp(
                        op=ast.Add(),
                        left=node.right,
                        right=ast.Constant(value=self.plus),
                    ),
                )
            case _:
                result = node
        return self.generic_visit(result)


def test_mult_right_plus():

    @MultRightPlus(plus=1).eval
    def result():
        1 + 3 * 4 - 5 * 2

    assert result == (1 + 3 * 5 - 5 * 3)


class MultLeftPlus(Macro):

    def __init__(self):
        super().__init__()

    def visit_Root(self, node):
        self.plus = node.attribute["plus"]
        return self.generic_visit(node)

    def visit_BinOp(self, node):
        match node.op:
            case ast.Mult():
                result = ast.BinOp(
                    op=node.op,
                    left=ast.BinOp(
                        op=ast.Add(),
                        left=node.left,
                        right=ast.Constant(value=self.plus),
                    ),
                    right=node.right,
                )
            case _:
                result = node
        return self.generic_visit(result)


def test_mult_left_plus():

    @MultLeftPlus().eval
    def result(plus=1):
        1 + 3 * 4 - 5 * 2

    assert result == (1 + 4 * 4 - 6 * 2)
