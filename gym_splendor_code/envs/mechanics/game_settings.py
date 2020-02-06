"""This file contains constants that are used in the game."""
from gym_splendor_code.envs.mechanics.enums import *

USE_TKINTER = True
USE_FAST_ACTION_GENERATOR = True
USE_TQDM = True
USE_TENSORFLOW_GPU = False
USE_LOCAL_TF = False

MAX_GEMS_IN_ONE_MOVE = 3
MAX_GEMS_ON_HAND = 10
MAX_RESERVED_CARDS = 3
MAX_CARDS_ON_BORD = 12
MAX_CARDS_IN_A_ROW_ON_BOARD = 4
POINTS_TO_WIN = 15

CARDS_IN_DECK = 90
CARDS_IN_ROWS = {Row.CHEAP : 40, Row.MEDIUM : 30, Row.EXPENSIVE : 20}

NOBLE_VICTORY_POINTS = 3
NOBLES_IN_DECK = 10
NOBLES_ON_BOARD_INITIAL = 3
MAX_NOBLES_IN_HAND = 3

MAX_NUMBER_OF_MOVES = 120

INITIAL_GEMS_ON_BOARD_DICT = {gem_color : 4 if  gem_color != GemColor.GOLD else 5 for gem_color in GemColor}
