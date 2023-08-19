# easy-ast

This package `easy-ast` contains several utility about AST transformer.

## Install

`pip install easy-ast`.

## Documents

### `AstDecorator`

This example exchange all plus and mult.

```python
import ast
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


@PlusMultExchange
def add(a, b):
    return a + b

assert add(2, 3) == 6
```
