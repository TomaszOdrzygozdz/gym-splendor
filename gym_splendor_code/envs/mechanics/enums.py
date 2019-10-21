"""This file contains enums that are used in the game."""

from enum import Enum, auto

class Row(Enum):
    """ Each card bolngs to one of 3 possible rows. Rows stand for prestige of card """
    CHEAP = 0
    MEDIUM = 1
    EXPENSIVE = 2


class GemColor(Enum):
    """
    This class represents colors of gems that are used in the game.
    """
    # We do not use auto() to have control of how gem colors are encoded to vectors (to make this encoding consistent).
    GOLD = 0
    RED = 1
    GREEN = 2
    BLUE = 3
    WHITE = 4
    BLACK = 5


class MoveType(Enum):
    """
    This class represents four main types of moves.
    """
    # We do not use auto() to have control of how move types are encoded to vectors (to make this encoding consistent).
    BUY_CARD = 0
    BUY_RESERVED_CARD = 1
    MODIFY_COINS = 2
    RESERVE_CARD = 3