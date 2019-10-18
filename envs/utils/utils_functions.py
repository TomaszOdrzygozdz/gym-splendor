from typing import Tuple

from envs.mechanics.gems_collection import GemsCollection
from envs.mechanics.players_hand import PlayersHand, GemColor

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

