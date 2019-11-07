from typing import Tuple, Set

from gym_splendor_code.envs.mechanics.card import Card
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.players_hand import GemColor

import numpy as np
from gym_splendor_code.envs.mechanics.game_settings import *

def tuple_of_gems_to_gems_collection(tuple_of_gems: Tuple[GemColor]) -> GemsCollection:
    """Return a gems collection constructed from the tuple of gems:
    Parameters:
     _ _ _ _ _ _
        tuple_of_gems: Tuple of gems (with possible repetitions).

    Returns: A gems collections. Example:
    (red, red, blue, green) is transformed to GemsCollection({red:2, blue:1, green:1, white:0, black:0, gold:0})."""
    gems_dict = {gem_color: 0 for gem_color in GemColor}
    for element in tuple_of_gems:
        gems_dict[element] += 1
    return GemsCollection(gems_dict)

def vectorize(vector):
    x = eval(vector.replace("NULL", "set()"))
    output = []
    for i in x.keys():
        if isinstance(x[i], dict):
            for j in x[i].keys():
                if "card" in j:
                    output.append([1 if y in x[i][j] else 0 for y in np.arange(CARDS_IN_DECK)])
                elif "noble" in j:
                    output.append([1 if y in x[i][j] else 0 for y in np.arange(NOBLES_IN_DECK)])
                elif "gems" in j:
                    output.append(x[i][j])

    return output
