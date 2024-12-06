# Individual
The Individual class represents an individual in the experiment.

## Concrete Individual classes
### Tree
Subclass of Individual, used for GP experiments.
Representation is a list of TreeNodes. Each node is either a function (inner node) or a terminal (leaf node). Tree nodes are in prefix order.

For instance, the function x*(y+1) is represented as:
```python
[
    FunctionNode(f_mul),
    TerminalNode('x'),
    FunctionNode(f_add),
    TerminalNode('y'),
    TerminalNode(1)
]
```

Converting a Tree to str generates a python function representing the tree:
```python
>>> from eckity.genetic_encodings.gp import Tree, TerminalNode, FunctionNode
>>> from eckity.base.untyped_functions import f_mul, f_add
>>> t1 = Tree([FunctionNode(f_mul), TerminalNode('x'), FunctionNode(f_add), TerminalNode('y'), TerminalNode(1)], terminal_set=['x', 'y'], function_set=[f_mul, f_add])
>>> print(str(t1))
def func_3(x, y):
  return f_mul(
    x,
    f_add(
      y,
      1
    )
  )
```

### Vector
Subclass of Individual, used for GP experiments.
The genotype is kept in the `vector` field.
The type of the list is homogenous and defined by the concrete vector class:

#### BitStringVector
Binary vector.

#### IntVector
Integer vector.

#### FloatVector
Float vector.
