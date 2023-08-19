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
from __future__ import annotations
import ast
import types
import typing

from .utility import *


class Einsum(Macro):

    def visit_Root(self, node):
        self.attribute = getattr(node, "attribute", {})
        self.dummy_index = self.attribute.get("_args", [])
        return self.generic_visit(node)

    def is_dummy_index(self, node):
        match node:
            case ast.Name(id=name):
                return name in self.dummy_index
            case _:
                return False

    def contain_dummy_index(self, node):
        return any(self.is_dummy_index(i) for i in ast.walk(node))

    def __init__(self, einsum=None):
        super().__init__()
        if einsum is None:
            einsum = ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="einsum", ctx=ast.Load())
        self.einsum = einsum

    def parse_tensor(self, node):
        match node:
            case ast.Subscript(value=value, slice=slices):
                match slices:
                    case ast.Tuple(elts=elts):
                        indices = elts
                    case _ as elt:
                        indices = [elt]
                dummy_shape = []
                result_indices = []
                for index in indices:
                    if self.is_dummy_index(index):
                        dummy_shape.append(index.id)
                        result_indices.append(ast.Slice())
                    else:
                        result_indices.append(index)
                result = ast.Subscript(value=value, slice=ast.Tuple(elts=result_indices), ctx=node.ctx)
                result.dummy_shape = dummy_shape
                return result
            case _:
                raise RuntimeError("Logical Critical Error")

    def parse_expr(self, node):
        match node:
            case ast.BinOp(op=op, left=left, right=right):
                parsed_left = self.parse_expr(left)
                parsed_right = self.parse_expr(right)
                result = ast.BinOp(op=op, left=parsed_left, right=parsed_right)
                if hasattr(parsed_left, "dummy_shape"):
                    if hasattr(parsed_right, "dummy_shape"):
                        match op:
                            case ast.Mult():
                                result_dummy = ([i for i in parsed_left.dummy_shape if i not in parsed_right.dummy_shape] +  #
                                                [i for i in parsed_right.dummy_shape if i not in parsed_left.dummy_shape])
                                string = "".join(parsed_left.dummy_shape) + "," + "".join(parsed_right.dummy_shape) + "->" + "".join(result_dummy)
                                result = ast.Call(
                                    func=self.einsum,
                                    args=[ast.Constant(value=string), parsed_left, parsed_right],
                                    keywords=[],
                                )
                                result.dummy_shape = result_dummy
                                return result
                            case ast.Add() | ast.Sub():
                                string = "".join(parsed_left.dummy_shape) + "->" + "".join(parsed_right.dummy_shape)
                                result = ast.BinOp(
                                    op=op,
                                    left=ast.Call(func=self.einsum, args=[
                                        ast.Constant(value=string),
                                        parsed_left,
                                    ], keywords=[]),
                                    right=parsed_right,
                                )
                                result.dummy_shape = parsed_right.dummy_shape
                                return result
                            case _:
                                raise NotImplementedError("This bin op not implemented")
                    else:
                        result.dummy_shape = parsed_left.dummy_shape
                        return result
                else:
                    if hasattr(parsed_right, "dummy_shape"):
                        result.dummy_shape = parsed_right.dummy_shape
                        return result
                    else:
                        return result
            case ast.UnaryOp(op=op, operand=operand):
                parsed_operand = self.parse_expr(operand)
                tensor_operand = hasattr(operand, "dummy_shape")
                result = ast.UnaryOp(op=op, operand=parsed_operand)
                if hasattr(parsed_operand, "dummy_shape"):
                    result.dummy_shape = parsed_operand.dummy_shape
                    return result
            case ast.Subscript() as subscript:
                if self.contain_dummy_index(subscript):
                    return self.parse_tensor(subscript)
            case _:
                pass
        return node

    def visit_Assign(self, node):
        # If targets length is 1 and the assignment contain dummy index
        # Then it is needed to be transformed
        if len(node.targets) == 1 and any(self.is_dummy_index(i) for i in ast.walk(node)):
            # ein_node: target = ein_expr
            # target: name | tensor                                  // target maybe a tensor or scalar
            # tensor: name[index] | name[tuple[index]]
            # index: dummy_index | expr                              // dummy_index or free_index
            # ein_expr: binop(op ein_expr ein_expr) | uniop(op ein_expr ) | tensor | expr
            # A fact: contain dummy && is subscript -> is tensor
            match node.targets[0]:
                case ast.Subscript() as target:
                    if self.contain_dummy_index(target):
                        parsed_target = self.parse_tensor(target)
                        parsed_expr = self.parse_expr(node.value)
                        string = "".join(parsed_expr.dummy_shape) + "->" + "".join(parsed_target.dummy_shape)
                        return ast.Assign(
                            targets=[parsed_target.value if len(parsed_target.slice.elts) == len(parsed_target.dummy_shape) else parsed_target],
                            value=ast.Call(func=self.einsum, args=[
                                ast.Constant(string),
                                parsed_expr,
                            ], keywords=[]),
                        )
                    else:
                        # Target is scalar
                        pass
                case _:
                    # Target is scalar
                    pass
            raise NotImplementedError()
        else:
            return self.generic_visit(node)
