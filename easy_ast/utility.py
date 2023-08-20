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
import textwrap
import inspect
import types
import typing


def _get_default_context(globals: typing.Optional[dict] = None, locals: typing.Optional[dict] = None) -> tuple[dict, dict]:
    """
    Get the context to eval/exec code, since it will get the second frame above the current frame, which is usually where user is.

    ---------------------
    | Target User Frame |
    ---------------------
    |   Library Frame   |
    ---------------------
    |   This function   |
    ---------------------
    """
    caller: types.FrameType = inspect.stack()[2].frame
    if globals is None:
        globals = caller.f_globals
    if locals is None:
        locals = caller.f_locals
    return globals, locals


def _attribute_from_function(func: ast.FunctionDef, globals: dict, locals: dict) -> dict[str, typing.Any]:
    """
    In Easy AST, function is used to create AST tree, so arguments could be used to record some user defined attribute.
    This function collect them and these attributes will be saved into the root node of AST tree.
    """
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
    """
    AST miss the code location information, which it is required to unparse AST tree, which is annoying, so wrap these two function into one.
    """
    return ast.unparse(ast.fix_missing_locations(tree))


def Statements(func: types.FunctionType, globals: typing.Optional[dict] = None, locals: typing.Optional[dict] = None) -> ast.Module:
    """
    This is used to get AST tree of statements, limited by python syntax, user could only use in the following way.

    @Statements
    def result(*args, **kwargs):
        stmt1
        stmt2
        stmt3
        ...

    This will result into this pseudo code
    result = AST(
        stmt1
        stmt2
        stmt3
        ....
    )

    Meanwhile, args and kwargs will be recorded into the root node(ast.Module) in the result.
    """
    globals, locals = _get_default_context(globals, locals)
    source: str = textwrap.dedent(inspect.getsource(func))
    module: ast.Module = ast.parse(source, mode="exec")
    assert isinstance(module.body[0], ast.FunctionDef), "Logical critical error"
    function_def: ast.FunctionDef = module.body[0]
    return ast.Module(body=function_def.body, type_ignores=module.type_ignores, attribute=_attribute_from_function(function_def, globals, locals))


def Expression(func: types.FunctionType, globals: typing.Optional[dict] = None, locals: typing.Optional[dict] = None) -> ast.Expression:
    """
    This is used to get AST tree of a single expression, limited by python syntax, user could only use in the following way.

    @Expression
    def result(*args, **kwargs):
        expression

    This will result into this pseudo code
    result = AST(
        expression
    )

    Meanwhile, args and kwargs will be recorded into the root node(ast.Expression) in the result.
    """
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
    """
    exec equivalent for AST tree.
    """
    globals, locals = _get_default_context(globals, locals)
    code: str = _ast_to_code(module)
    exec(code, globals, locals)


def Eval(expression: ast.Expression, globals=None, locals=None) -> typing.Any:
    """
    eval equivalent for AST tree.
    """
    globals, locals = _get_default_context(globals, locals)
    code: str = _ast_to_code(expression)
    return eval(code, globals, locals)


class Macro(ast.NodeTransformer):
    """
    This is used to define a macro. To create a macro, user need to create a subtype of this class.

    class UserDefinedMacro(Macro):
       ...

    To call the macro, call the instance directly or use decorator followed with Statements or Expression

    1. UserDefinedMacro()(ast.parse("..."))

    2. @UserDefinedMacro()
       @Expression
       def result():
           ...

    User may want to save data during process the AST transformer, which could just be stored in the instance of the macro class.

    Since Expression and Statements would save extra attribute to root node, this class provide a function called visit_Root to visit the root node,
    to avoid duplicated code in both visit_Module(return by Statements) and visit_Expression(return by Expression).

    Sometimes user may want to eval/exec code right after macro process done. class Macro provide two function Macro.eval and Macro.exec.

    @Eval
    @UserDefinedMacro()
    @Expression
    def result():
        ...

    could be simplified as

    @UserDefinedMacro().eval
    def result():
        ...
    """

    def __call__(self: typing.Self, tree: ast.AST) -> ast.AST:
        """
        Run the AST transformer over an AST tree
        """
        return self.visit(tree)

    def exec(self: typing.Self, func: types.FunctionType, globals: typing.Optional[dict] = None, locals: typing.Optional[dict] = None) -> None:
        """
        Extract the statements in the given function, process with the macro and exec in time.
        """
        globals, locals = _get_default_context(globals, locals)
        module: ast.Module = Statements(func)
        result: ast.Module = typing.cast(ast.Module, self(module))
        Exec(result, globals, locals)

    def eval(self: typing.Self, func: types.FunctionType, globals: typing.Optional[dict] = None, locals: typing.Optional[dict] = None) -> typing.Any:
        """
        Extract the expression in the given function, process with the macro and eval in time.
        """
        globals, locals = _get_default_context(globals, locals)
        module: ast.Module = Expression(func)
        result: ast.Expression = typing.cast(ast.Expression, self(module))
        return Eval(result, globals, locals)

    def visit_Module(self: typing.Self, node: ast.Module) -> ast.Module:
        return typing.cast(ast.Module, self.visit_Root(node))

    def visit_Expression(self: typing.Self, node: ast.Expression) -> ast.Expression:
        return typing.cast(ast.Expression, self.visit_Root(node))

    def visit_Root(self: typing.Self, node: ast.Module | ast.Expression) -> ast.Module | ast.Expression:
        """
        The default visitor for module and expression. User could override this function once to override visitor for both module and expression.

        The way AST visitor in python AST transformer follows these rules.
        1. try to run visit for the specific node such as visit_Name, visit_Tuple
        2. if specific node not defined, use the default visit, which do nothing but call generic_visit
        3. generic_visit is defined to call all child node recursively, this function should not be overrided.

        So, if user want to walk in the AST tree, after/before node type specific code, calling genenric_visit after is usually required.
        """
        return typing.cast(ast.Module | ast.Expression, self.generic_visit(node))


class AstDecorator(ast.NodeTransformer):
    """
    This is used to create AST transformer decorator. The usage is to create a subtype of this class and decorate function in this way:

    @xxx
    @UserDefinedDecorator()
    @xxx
    def function(...):
        ...

    All AST tree in the function will be transformed based on visitor function defined in subtype of this class.

    Note:
    Because of limit in python syntax, multiple AstDecorators of the same type could not be used in decorator form.
    If user want to implement the behavior, call the decorator manually like this

    def function(...):
        ...
    function = xxx(function)
    function = UserDefinedDecorator(aaa)(function)
    function = UserDefinedDecorator(bbb)(function)
    function = yyy(function)
    """

    def __call__(self: typing.Self, func: types.FunctionType) -> types.FunctionType:
        """
        Transform AST tree of the given function and return.
        """
        globals, locals = _get_default_context(None, None)
        source: str = textwrap.dedent(inspect.getsource(func))
        module: ast.Module = ast.parse(source, mode="exec")
        assert isinstance(module.body[0], ast.FunctionDef), "Logical critical error"
        function_def: ast.FunctionDef = module.body[0]
        function_def_without_decorator_above: ast.FunctionDef = self._remove_decorator_above(function_def)
        new_function_def: ast.Module = self.visit(function_def_without_decorator_above)
        new_source: str = _ast_to_code(new_function_def)
        exec(new_source, globals, locals)
        return locals[func.__name__]

    def __init__(self: typing.Self) -> None:
        super().__init__()
        self.globals, self.locals = _get_default_context()

    def _remove_decorator_above(self: typing.Self, node: ast.FunctionDef) -> ast.FunctionDef:
        """
        Remove the decorator above the current decorator.

        Since maybe this decorator is stored in other variable. So it is needed to eval every decorator to check which one is self.
        However we can not ensure it is this decorator if the type is the same to this. So do not apply multiple decorators of the same type in decorator form.
        """
        decorator_list: list[ast.expr] = []
        meet_me: bool = False
        for decorator in node.decorator_list:
            if meet_me:
                decorator_list.append(decorator)
            decorator_value = Eval(ast.Expression(body=decorator), self.globals, self.locals)
            # Even we have the same type, we may be different object.
            # But it is impossible to check, so currently, just check whether the types are the same.
            if isinstance(decorator_value, type(self)):
                meet_me = True
        result: ast.FunctionDef = ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            returns=node.returns,
            type_comment=node.type_comment,
            decorator_list=decorator_list,
        )
        return result
