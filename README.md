# easy-ast

This package `easy-ast` contains several utility about AST transformer.

## Install

`pip install easy-ast`.

## Documents

### `AstDecorator`

This example exchange all plus and mult.

```python
import ast
from easy_ast import *


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


assert add(2, 3) == 6
```

### Get AST instance

Provide `Statements` and `Expression` to get the AST instance of code.
`Exec` and `Eval` is used to exec/eval code directly on AST

```python
import ast
from easy_ast import *


@Expression
def tree():
    (x + 1)**2


assert isinstance(tree, ast.Expression)

x = 6
assert Eval(tree) == 49
```

Because of [the limit of python](https://docs.python.org/3/library/functions.html#exec),
executing code will not effect variable in time, which may be solved in the future,
see [pep558](https://peps.python.org/pep-0558/) and [pep667](https://peps.python.org/pep-0667/).
Currently, the only way to use variable updated in exec is to visit it via `locals()` manually,
or use it inside exec content directly.

```python
import ast
from easy_ast import *


@Statements
def tree():
    y = x + 1
    z = y * 2
    assert z == 6


assert isinstance(tree, ast.Module)

x = 2
Exec(tree)
assert locals()["z"] == 6
```

### Macro

Class `Macro` is used to create macro. Here is a simple example to do operator directly on python list.

```python
import ast
from easy_ast import *


class List(Macro):

    def __init__(self):
        super().__init__()
        self.symbol_current = 0

    def visit_BinOp(self, node):
        i = f"__list_loop_variable_{self.symbol_current}"
        self.symbol_current += 1
        j = f"__list_loop_variable_{self.symbol_current}"
        self.symbol_current += 1
        # Return [op(left, right) for i,j in zip(left, right)]
        return ast.ListComp(
            elt=ast.BinOp(
                op=node.op,
                left=ast.Name(id=i, ctx=ast.Load()),
                right=ast.Name(id=j, ctx=ast.Load()),
            ),
            generators=[
                ast.comprehension(
                    target=ast.Tuple(elts=[
                        ast.Name(id=i, ctx=ast.Store()),
                        ast.Name(id=j, ctx=ast.Store()),
                    ]),
                    iter=ast.Call(
                        func=ast.Name(id="zip", ctx=ast.Load()),
                        args=[
                            self.generic_visit(node.left),
                            self.generic_visit(node.right),
                        ],
                        keywords=[],
                    ),
                    ifs=[],
                    is_async=0,
                )
            ],
        )


a = [1, 2, 3]
b = [1, 2, 3]


@List().eval
def c():
    a * b


assert c == [1, 4, 9]
```

### Tensor contract

This repository implements an AST transformer for Einstein notation based on `Macro` for numpy array.

```python
import numpy as np
from easy_ast.tensor_contract import TensorContract

b = np.array([1, 2])
c = np.array([1, 2])
expect = -np.array([[1, 2], [2, 4]])


@TensorContract().exec
def _(i, j):
    a[i, j] = -b[i] * c[j]
    assert np.all(a == expect)


a = np.random.randn(3, 2, 6, 5)
b = np.random.randn(3, 4)
c = np.random.randn(5, 4, 2)
d = np.einsum("ijk,il,klm->mj", a[:, 0], b, c)
assert d.shape == (2, 6)
r = np.zeros([6, 2, 2])


@TensorContract().exec
def _(i, j, k, l, m):
    # i3, j6, k5, l4, m2
    r[j, m, 1] = a[i, 0, j, k] * b[i, l] * c[k, l, m] - d[m, j]
    assert np.sum(np.abs(r)) < 1e-6
```

It also supports non-standard Einstein notation.

```python
import numpy as np
from easy_ast.tensor_contract import TensorContract

a = np.array([1, 2])
b = np.array([1, 2])


@TensorContract().exec
def _(i):
    c[i] = a[i] * b[i]
    assert np.all(c == [1, 4])
    d = a[i]
    assert d == 3
```
