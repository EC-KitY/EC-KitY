![image](https://user-images.githubusercontent.com/62753120/163423530-1c85e43f-48a9-4fbd-827e-f97a1f174db0.png)
![PyPI](https://img.shields.io/pypi/v/eckity)


**EC-KitY** is a Python tool kit for doing evolutionary computation, and it is scikit-learn compatible.

Currently we have implemented Genetic Algorithm (GA) and tree-based Genetic Programming (GP), but EC-KitY will grow!

**EC-KitY** is:
- A comprehensive toolkit for running evolutionary algorithms
- Written in Python
- Can work with or without scikit-learn, i.e., supports both sklearn and non-sklearn modes
- Designed with modern software engineering in mind
- Designed to support all popular EC paradigms (GA, GP, ES, coevolution, multi-objective, etc').

### Dependencies
For the basic evolution mode, EC-KitY requires:
- Python (>=3.6)
- Numpy (>=1.14.6)
- Pandas (>=0.25.0)
- Overrides (>= 6.1.0)

For sklearn mode, EC-KitY additionally requires:
- scikit-learn (>=0.24.2)

### User installation

`pip install eckity`

### Documentation

API is available [here](https://api.eckity.org)

(Work in progress - some modules and functions are not documented yet.)

### Tutorials
The tutorials are available [here](https://github.com/EC-KitY/EC-KitY/wiki/Tutorials), walking you through running EC-KitY both in sklearn mode and in non-sklearn mode.

### Examples
More examples are in the [examples](https://github.com/EC-KitY/EC-KitY/tree/main/examples "examples") folder.
All you need to do is define a fitness-evaluation method, through a `SimpleIndividualEvaluator` sub-class.
You can run the examples with ease by opening this [colab notebook](https://colab.research.google.com/drive/1mpr3EGb1rpoK-_zugszQkv1sWVm-ZQiB?usp=sharing).

### Basic example (no sklearn)
You can run an EA with just 3 lines of code. The problem being solved herein is simple symbolic regression.

Additional information on this problem can be found in the [Symbolic Regression Tutorial](https://github.com/EC-KitY/EC-KitY/wiki/Tutorial:-Symbolic-Regression).
```python
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.subpopulation import Subpopulation
from examples.treegp.non_sklearn_mode.symbolic_regression.sym_reg_evaluator import SymbolicRegressionEvaluator

algo = SimpleEvolution(Subpopulation(SymbolicRegressionEvaluator()))
algo.evolve()
print(f'algo.execute(x=2,y=3,z=4): {algo.execute(x=2, y=3, z=4)}')
```

### Example with sklearn

The problem being solved herein is the same problem, but in this case we also involve sklearn compatability - a core feature of EC-KitY.
Additional information for this example can be found in the [Sklearn Symbolic Regression Tutorial](https://github.com/EC-KitY/EC-KitY/wiki/Tutorial:-Sklearn-Compatible-Symbolic-Regression).

A simple sklearn-compatible EA run:

```python
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.creators.gp_creators.full import FullCreator
from eckity.genetic_encodings.gp.tree.utils import create_terminal_set
from eckity.sklearn_compatible.regression_evaluator import RegressionEvaluator
from eckity.sklearn_compatible.sk_regressor import SKRegressor
from eckity.subpopulation import Subpopulation

X, y = make_regression(n_samples=100, n_features=3)
terminal_set = create_terminal_set(X)

algo = SimpleEvolution(Subpopulation(creators=FullCreator(terminal_set=terminal_set),
                                     evaluator=RegressionEvaluator()))
regressor = SKRegressor(algo)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
regressor.fit(X_train, y_train)
print('MAE on test set:', mean_absolute_error(y_test, regressor.predict(X_test)))
```

### Feature comparison
Here's a comparison table. The full paper is available [here](https://arxiv.org/abs/2207.10367).
![image](https://github.com/EC-KitY/EC-KitY/blob/main/features.JPG?raw=true)

### Authors
[Moshe Sipper](http://www.moshesipper.com/ "Moshe Sipper"), 
[Achiya Elyasaf](https://achiya.elyasaf.net/ "Achiya Elyasaf"),
[Itai Tzruia](https://www.linkedin.com/in/itai-tzruia-4a47a91b8/),
Tomer Halperin

### Citation

Citations are always appreciated 😊:
```
@article{eckity2023,
author = {Moshe Sipper and Tomer Halperin and Itai Tzruia and Achiya Elyasaf},
title = {{EC-KitY}: Evolutionary computation tool kit in {Python} with seamless machine learning integration},
journal = {SoftwareX},
volume = {22},
pages = {101381},
year = {2023},
url = {https://www.sciencedirect.com/science/article/pii/S2352711023000778},
}

@misc{eckity2022git,
    author = {Sipper, Moshe and Halperin, Tomer and Tzruia, Itai and  Elyasaf, Achiya},
    title = {{EC-KitY}: Evolutionary Computation Tool Kit in {Python}},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://www.eckity.org/} }
}

```

### Sample repos using EC-KitY
- [EC-KitY-Maze-Example](https://github.com/RonMichal/EC-KitY-Maze-Example/tree/maze_example/examples/vectorga/maze)
- [EvolutionTSP](https://github.com/nogazax/EvolutionTSP)
- [Solving The 'Nurse Scheduling Problem' With EC-KitY](https://github.com/harelaf/Nurse-Scheduling-Problem)





