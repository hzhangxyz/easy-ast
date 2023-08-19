#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Hao Zhang<zh970205@mail.ustc.edu.cn>
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
"""Easy AST"""
from __future__ import annotations

import ast
import textwrap
import inspect
import types
import typing

__version__: str = "0.0.1"


def _get_default_context(globals: typing.Optional[dict] = None, locals: typing.Optional[dict] = None) -> tuple[dict, dict]:
    caller: types.FrameType = inspect.stack()[2].frame
    if globals is None:
        globals = caller.f_globals
    if locals is None:
        locals = caller.f_locals
    return globals, locals


def _attribute_from_function(func: ast.FunctionDef, globals: dict, locals: dict) -> dict[str, typing.Any]:
    assert len(func.args.posonlyargs) == 0, "Position only argument not allowed here"
    assert len(func.args.kwonlyargs) == 0, "Keyword only argument not allowed here"
    keys: list[str] = []
    for key in func.args.args:
        keys.append(key.arg)
    values: list[typing.Any] = []
    for value in func.args.defaults:
        # eval arg and append to values right now
        values.append(eval(_ast_to_code(value), globals, locals))
    args_number: int = len(keys) - len(values)
    result: dict[str, typing.Any] = dict(zip(keys[args_number:], values))
    result["_args"] = keys[:args_number]
    return result


def _ast_to_code(tree: ast.AST) -> str:
    return ast.unparse(ast.fix_missing_locations(tree))


def Statements(func: types.FunctionType, globals: typing.Optional[dict] = None, locals: typing.Optional[dict] = None) -> ast.Module:
    globals, locals = _get_default_context(globals, locals)
    source: str = textwrap.dedent(inspect.getsource(func))
    module: ast.Module = ast.parse(source, mode="exec")
    assert isinstance(module.body[0], ast.FunctionDef), "Logical critical error"
    function_def: ast.FunctionDef = module.body[0]
    return ast.Module(body=function_def.body, type_ignores=module.type_ignores, attribute=_attribute_from_function(function_def, globals, locals))


def Expression(func: types.FunctionType, globals: typing.Optional[dict] = None, locals: typing.Optional[dict] = None) -> ast.Expression:
    globals, locals = _get_default_context(globals, locals)
    source: str = textwrap.dedent(inspect.getsource(func))
    module: ast.Module = ast.parse(source, mode="exec")
    assert isinstance(module.body[0], ast.FunctionDef), "Logical critical error"
    function_def: ast.FunctionDef = module.body[0]
    assert len(function_def.body) == 1, "Expression function should contain only a single expression"
    assert isinstance(function_def.body[0], ast.Expr), "The only statement in Expression should be an expression"
    expression: ast.expr = function_def.body[0].value
    return ast.Expression(body=expression, attribute=_attribute_from_function(function_def, globals, locals))


def Exec(module: ast.Module, globals: typing.Optional[dict] = None, locals: typing.Optional[dict] = None) -> None:
    globals, locals = _get_default_context(globals, locals)
    code: str = _ast_to_code(module)
    exec(code, globals, locals)


def Eval(expression: ast.Expression, globals=None, locals=None) -> typing.Any:
    globals, locals = _get_default_context(globals, locals)
    code: str = _ast_to_code(expression)
    return eval(code, globals, locals)


class Macro(ast.NodeTransformer):

    def __new__(cls: type[Macro], tree: typing.Optional[ast.AST] = None):
        if tree is None:
            return super().__new__(cls)
        else:
            obj: Macro = cls()
            return obj.visit(tree)

    @classmethod
    def exec(cls: type[Macro], func: types.FunctionType, globals: typing.Optional[dict] = None, locals: typing.Optional[dict] = None) -> None:
        globals, locals = _get_default_context(globals, locals)
        module: ast.Module = Statements(func)
        result: ast.Module = typing.cast(ast.Module, cls(module))
        Exec(result, globals, locals)

    @classmethod
    def eval(cls: type[Macro], func: types.FunctionType, globals: typing.Optional[dict] = None, locals: typing.Optional[dict] = None) -> typing.Any:
        globals, locals = _get_default_context(globals, locals)
        module: ast.Module = Statements(func)
        result: ast.Expression = typing.cast(ast.Expression, cls(module))
        return Eval(result, globals, locals)

    def visit_Module(self: typing.Self, node: ast.Module) -> ast.Module:
        return typing.cast(ast.Module, self.visit_Root(node))

    def visit_Expression(self: typing.Self, node: ast.Expression) -> ast.Expression:
        return typing.cast(ast.Expression, self.visit_Root(node))

    def visit_Root(self: typing.Self, node: ast.Module | ast.Expression) -> ast.Module | ast.Expression:
        return typing.cast(ast.Module | ast.Expression, self.generic_visit(node))


class AstDecorator(ast.NodeTransformer):

    def __new__(cls: type[AstDecorator], func: typing.Optional[types.FunctionType] = None):
        if func is None:
            return super().__new__(cls)
        else:
            globals, locals = _get_default_context(None, None)
            source: str = textwrap.dedent(inspect.getsource(func))
            tree: ast.Module = ast.parse(source, mode="exec")
            new_tree: ast.Module = cls().visit(tree)
            new_source: str = _ast_to_code(new_tree)
            local: dict = {}
            exec(new_source, globals, locals)
            return locals[func.__name__]

    def __init__(self: typing.Self) -> None:
        super().__init__()
        self._never_function: bool = True

    def visit_FunctionDef(self: typing.Self, node: ast.FunctionDef) -> ast.AST:
        result: ast.FunctionDef
        if self._never_function:
            self._never_function = False
            decorator_list: list[ast.expr] = []
            meet_me: bool = False
            for decorator in node.decorator_list:
                if meet_me:
                    decorator_list.append(decorator)
                if isinstance(decorator, ast.Name):
                    if decorator.id == type(self).__name__:
                        meet_me = True
            result = ast.FunctionDef(
                name=node.name,
                args=node.args,
                body=node.body,
                returns=node.returns,
                type_comment=node.type_comment,
                decorator_list=decorator_list,
            )
        else:
            result = node
        return self.generic_visit(result)
