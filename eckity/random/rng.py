from abc import ABC, abstractmethod
from typing import Union


class RNG(ABC):
    def __init__(self, seed: Union[int, float]):
        self._seed = seed

    def generation_seed(self, seed):
        pass

    @abstractmethod
    def random_seed(self, seed: Union[int, float]):
        pass

    @abstractmethod
    def randint(self, a: int, b: int):
        pass
