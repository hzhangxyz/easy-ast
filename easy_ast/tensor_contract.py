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


class TensorContract(Macro):

    def visit_Root(self: typing.Self, node: ast.Module | ast.Expression) -> ast.Module | ast.Expression:
        """
        Get the dummy_index set from the attribute recorded in root node.
        """
        attribute = getattr(node, "attribute", {})
        self.dummy_index |= set(attribute.get("_args", []))
        return typing.cast(ast.Module | ast.Expression, self.generic_visit(node))

    def _is_dummy_index(self: typing.Self, node: ast.AST) -> bool:
        """
        Check whether a node is dummy index node
        """
        match node:
            case ast.Name(id=name):
                return name in self.dummy_index
            case _:
                return False

    def _contain_dummy_index(self: typing.Self, node: ast.AST) -> bool:
        """
        Check whether a node contains dummy index node
        """
        return any(self._is_dummy_index(i) for i in ast.walk(node))

    @staticmethod
    def _numpy_contract(tensor_1: ast.AST, tensor_2: ast.AST, result_free_index: list[str]) -> ast.AST:
        """
        The default contract function, it will create AST to contract two tensor, tensor dummy shape is store inside the node.
        The result reterned is an AST with free index information stored.
        """
        tensor_1_free_index: list[str] = getattr(tensor_1, "free_index")
        tensor_2_free_index: list[str] = getattr(tensor_2, "free_index")
        index_store: dict[str, str] = {}
        index_count: int = 0
        for index in tensor_1_free_index:
            if index not in index_store:
                index_store[index] = chr(ord('a') + index_count)
                index_count += 1
        for index in tensor_2_free_index:
            if index not in index_store:
                index_store[index] = chr(ord('a') + index_count)
                index_count += 1
        string: str = ("".join(index_store[index] for index in tensor_1_free_index) + "," + "".join(index_store[index] for index in tensor_2_free_index) + "->" +
                       "".join(index_store[index] for index in result_free_index))
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np"),
                attr="einsum",
                ctx=ast.Load(),
            ),
            args=[
                ast.Constant(value=string),
                tensor_1,
                tensor_2,
            ],
            keywords=[],
            free_index=result_free_index,
        )

    @staticmethod
    def _numpy_transpose(tensor: ast.AST, result_free_index: list[str]) -> ast.AST:
        """
        The default transpose function, it will create AST to transpose a tensor, tensor dummy shape is store inside the node.
        The result reterned is an AST with free index information stored.
        """
        tensor_free_index: list[str] = getattr(tensor, "free_index")
        index_store: dict[str, str] = {}
        index_count: int = 0
        for index in tensor_free_index:
            if index not in index_store:
                index_store[index] = chr(ord('a') + index_count)
                index_count += 1
        string: str = ("".join(index_store[index] for index in tensor_free_index) + "->" + "".join(index_store[index] for index in result_free_index))
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="np"),
                attr="einsum",
                ctx=ast.Load(),
            ),
            args=[
                ast.Constant(value=string),
                tensor,
            ],
            keywords=[],
            free_index=result_free_index,
        )

    def __init__(
        self: typing.Self,
        dummy_index: typing.Optional[set[str]] = None,
        contract: typing.Optional[typing.Callable[[ast.AST, ast.AST, list[str]], ast.AST]] = None,
        transpose: typing.Optional[typing.Callable[[ast.AST, list[str]], ast.AST]] = None,
    ) -> None:
        """
        Create the TensorContract object.
        """
        super().__init__()

        self.dummy_index: set[str] = set()
        if dummy_index is not None:
            self.dummy_index |= dummy_index
        self.contract: typing.Callable[[ast.AST, ast.AST, list[str]], ast.AST]
        if contract is None:
            self.contract = self._numpy_contract
        else:
            self.contract = contract
        self.transpose: typing.Callable[[ast.AST, list[str]], ast.AST]
        if transpose is None:
            self.transpose = self._numpy_transpose
        else:
            self.transpose = transpose

    def parse_tensor(self, node: ast.Subscript) -> ast.AST:
        """
        Convert the given raw tensor node of einsum form to the python form.

        This function is called only when the node is really a einsum form tensor.

        A einsum form tensor is:
        | Subscript(Name, Tuple[list[Index]])
        | Subscript(Name, Index)

        Index is:
        | Name with dummy index
        | any python expr

        Note:
        The target of the assignment of einsum form is:
        | einsum form tensor
        | Name
        This could be caused by the contract result become a scalar

        The parsed python form tensor is:
        | Subscript(Name) with free_index attr
        | Name with free_index attr

        It could be:
        |- Generated directly from einsum form
        |- Expr result
        |- Target
        All of these three could be contain free_index as empty list
        """
        match node:
            case ast.Subscript(value=value, slice=slices):
                match slices:
                    case ast.Tuple(elts=elts):
                        indices = elts
                    case _ as elt:
                        indices = [elt]
                free_index: list[str] = []
                result_index: list[ast.AST] = []
                for index in indices:
                    if self._is_dummy_index(index):
                        free_index.append(typing.cast(ast.Name, index).id)
                        result_index.append(ast.Slice())
                    else:
                        result_index.append(index)
                return ast.Subscript(
                    value=value,
                    slice=ast.Tuple(elts=result_index),
                    ctx=node.ctx,
                    free_index=free_index,
                )
            case _:
                raise RuntimeError("Logical Critical Error")

    def parse_expr(self: typing.Self, node: ast.AST) -> ast.AST:
        result: ast.AST
        match node:
            case ast.BinOp(op=op, left=left, right=right):
                parsed_left: ast.AST = self.parse_expr(left)
                parsed_right: ast.AST = self.parse_expr(right)
                result = ast.BinOp(op=op, left=parsed_left, right=parsed_right)
                if hasattr(parsed_left, "free_index"):
                    if hasattr(parsed_right, "free_index"):
                        match op:
                            case ast.Mult():
                                result_free_index = ([i for i in parsed_left.free_index if i not in parsed_right.free_index] +  #
                                                     [i for i in parsed_right.free_index if i not in parsed_left.free_index])
                                result = self.contract(parsed_left, parsed_right, result_free_index)
                                setattr(result, "free_index", result_free_index)
                                return result
                            case ast.Add() | ast.Sub():
                                result_free_index = getattr(parsed_right, "free_index")
                                result = ast.BinOp(
                                    op=op,
                                    left=self.transpose(parsed_left, result_free_index),
                                    right=parsed_right,
                                )
                                setattr(result, "free_index", result_free_index)
                                return result
                            case _:
                                raise NotImplementedError("This bin op not implemented")
                    else:
                        setattr(result, "free_index", getattr(parsed_left, "free_index"))
                        return result
                else:
                    if hasattr(parsed_right, "free_index"):
                        setattr(result, "free_index", getattr(parsed_right, "free_index"))
                        return result
                    else:
                        return result
            case ast.UnaryOp(op=op, operand=operand):
                parsed_operand = self.parse_expr(operand)
                tensor_operand = hasattr(operand, "free_index")
                result = ast.UnaryOp(op=op, operand=parsed_operand)
                if hasattr(parsed_operand, "free_index"):
                    setattr(result, "free_index", getattr(parsed_operand, "free_index"))
                    return result
            case ast.Subscript() as subscript:
                if self._contain_dummy_index(subscript):
                    return self.parse_tensor(subscript)
            case _:
                pass
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        # Macro process only if it is a assignment containing dummy index.
        if self._contain_dummy_index(node):
            assert len(node.targets) == 1, "The target of tensor contract assignment does not support unpacking"
            target: ast.AST = node.targets[0]
            parsed_target: ast.AST
            parsed_expr: ast.AST
            match target:
                case ast.Subscript() as target:
                    if self._contain_dummy_index(target):
                        # Target is a einsum tensor
                        parsed_target = self.parse_tensor(target)
                        assert isinstance(parsed_target, ast.Subscript)
                        assert isinstance(parsed_target.slice, ast.Tuple)
                        # The target could be a part of existent tensor or a new tensor
                        # if two length equal -> new tensor, else -> part of eixstent tensor
                        if len(parsed_target.slice.elts) == len(getattr(parsed_target, "free_index")):
                            setattr(parsed_target.value, "free_index", getattr(parsed_target, "free_index"))
                            parsed_target = parsed_target.value
                    else:
                        # Target is scalar
                        parsed_target = target
                        setattr(parsed_target, "free_index", [])
                case _:
                    # Target is scalar
                    parsed_target = target
                    setattr(parsed_target, "free_index", [])
            parsed_expr = self.parse_expr(node.value)
            return ast.Assign(
                targets=[parsed_target],
                value=self.transpose(parsed_expr, getattr(parsed_target, "free_index")),
            )
        else:
            return self.generic_visit(node)
