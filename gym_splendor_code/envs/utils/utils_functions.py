from typing import Tuple, Set

from gym_splendor_code.envs.mechanics.card import Card
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.players_hand import GemColor

import numpy as np
from gym_splendor_code.envs.mechanics.game_settings import *


def tuple_of_gems_to_gems_collection(tuple_of_gems: Tuple[GemColor], val = 1, return_val = [1], return_colors = set()) -> GemsCollection:
    """Return a gems collection constructed from the tuple of gems:
    Parameters:
     _ _ _ _ _ _
        tuple_of_gems: Tuple of gems (with possible repetitions).

    Returns: A gems collections. Example:
    (red, red, blue, green) is transformed to GemsCollection({red:2, blue:1, green:1, white:0, black:0, gold:0})."""
    gems_dict = {gem_color: 0 for gem_color in GemColor}
    for element in tuple_of_gems:
        gems_dict[element] += val
    for i, element in enumerate(return_colors):
        gems_dict[element] -= return_val[i]
    return GemsCollection(gems_dict)
