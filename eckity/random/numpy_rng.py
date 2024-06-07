from time import time
from typing import Optional, Union

import numpy as np

from eckity.random import RNG


class NumpyRNG(RNG):
    def __init__(
        self,
        seed: Union[int, float] = time(),
    ):
        super().__init__(seed)

    def seed(self, seed: Optional[Union[int, float]] = None):
        self._seed = np.random.seed(seed)

    def randint(self, a: int, b: int):
        return np.random.randint(a, b)
