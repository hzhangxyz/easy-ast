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
import functools
from easy_ast import AstDecorator


class PlusMultExchange(AstDecorator):

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            result = ast.BinOp(left=node.left, op=ast.Mult(), right=node.right)
        elif isinstance(node.op, ast.Mult):
            result = ast.BinOp(left=node.left, op=ast.Add(), right=node.right)
        else:
            result = node
        return self.generic_visit(result)


@PlusMultExchange()
def add(a, b):
    return a + b


def test_add():
    assert add(2, 3) == 6


@PlusMultExchange()
def func(a, b, c, d):
    return (a + 2) * b - (c + d)


def test_func():
    assert func(2, 3, 4, 5) == -13


def logger(func):

    @functools.wraps(func)
    def result(*args, **kwargs):
        global count
        count += 1
        return func(*args, **kwargs)

    return result


@logger
@PlusMultExchange()
def log_before_dec_add(a, b):
    return a + b


def test_log_before_dec_add():
    global count
    count = 0
    assert log_before_dec_add(2, 3) == 6
    assert count == 1


@PlusMultExchange()
@logger
def log_after_dec_add(a, b):
    return a + b


def test_log_after_dec_add():
    global count
    count = 0
    assert log_after_dec_add(2, 3) == 6
    assert count == 1
