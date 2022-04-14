<img src="https://github.com/moshesipper/scikit-gp/blob/master/doc/header.jpg" width="90%"/>

EC-KitY is a Python tool kit for doing evolutionary computation. 
It is scikit-learn-compatible and is distributed under the GNU General Public License v3.0.
Currently we have implemented tree-based genetic programming (GP), but EC-KitY will grow!

### Dependencies
For the basic evolution mode, EC-KitY requires:
- numpy (>=1.21.5)
- pandas (>=1.3.5)
- overrides (>= 6.1.0)

For sklearn mode, EC-KitY additionally requires:
- scikit-learn (>=0.24.2)

### User installation

`git clone https://github.com/EC-KitY/EC-KitY.git`

For basic package installation: `pip install -r base_requirements.txt`

For the extended package: `pip install -r base_requirements.txt -r extensions_requirements.txt`

### Documentation

is available [here](https://htmlpreview.github.io/?https://github.com/moshesipper/ec-kity/tree/master/doc "here"). 

### Examples
More examples are in the [examples](https://github.com/moshesipper/ec-kity/tree/master/examples "examples") folder.
All you need to do is define a fitness-evaluation method, through a `SimpleIndividualEvaluator` sub-class.

### Basic example (no sklearn)
You can run an EA with just 3 lines of code. The problem being solved herein is simple symbolic regression. [LINK TO TUTORIAL]  
```python
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.subpopulation import Subpopulation
from examples.treegp.non_sklearn_mode.symbolic_regression.sym_reg_evaluator import SymbolicRegressionEvaluator

algo = SimpleEvolution(Subpopulation(SymbolicRegressionEvaluator()))
algo.evolve()
print(f'algo.execute(x=2,y=3,z=4): {algo.execute(x=2, y=3, z=4)}')
```

### Example with sklearn
A simple sklearn-compatible EA run:

```python
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.creators.gp_creators.full import FullCreator
from eckity.genetic_encodings.gp.tree.utils import create_terminal_set
from eckity.sklearn_compatible.regression_evaluator import RegressionEvaluator
from eckity.sklearn_compatible.sk_regressor import SkRegressor
from eckity.subpopulation import Subpopulation

X, y = make_regression(n_samples=100, n_features=3)
terminal_set = create_terminal_set(X)

algo = SimpleEvolution(Subpopulation(creators=FullCreator(terminal_set=terminal_set),
                                     evaluator=RegressionEvaluator()))
regressor = SkRegressor(algo)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
regressor.fit(X_train, y_train)
print('MAE on test set:', mean_absolute_error(regressor.predict(X_test), y_test))
```

### Authors
[Moshe Sipper](http://www.moshesipper.com/ "Moshe Sipper"), 
[Achiya Elyasaf](https://achiya.elyasaf.net/ "Achiya Elyasaf"),
[Itai Tzruia](https://www.linkedin.com/in/itai-tzruia-4a47a91b8/),
Tomer Halperin

### Citation

If you wish to cite this work please use:
```
@misc{eckity2022,
  author = {Sipper, Moshe and Halperin, Tomer and Tzruia, Itai and  Elyasaf, Achiya},
  title = {EC-KitY: Evolutionary Computation Tool Kit in Python},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://eckity.org} }
}
```





