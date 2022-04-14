from sklearn.base import RegressorMixin

from eckity.sklearn_compatible.sklearn_wrapper import SklearnWrapper


class SkRegressor(SklearnWrapper, RegressorMixin):
    pass
