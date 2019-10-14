from typing import Dict

from abc import ABCMeta, abstractmethod
from envs.mechanics.enums import GemColor
from envs.mechanics.gems_collection import GemsCollection


class PreCard(metaclass=ABCMeta):
    '''This is an abstract class that shares features of both game card and noble card. Classes: Card and Noble inherits
    the attributes of PreCard.'''
    @abstractmethod
    def __init__(self,
                 name: str,
                 id: int,
                 price: GemsCollection,
                 victory_points: int) -> None:
        self.name = name
        self.id = id
        self.price = price
        self.victory_points = victory_points