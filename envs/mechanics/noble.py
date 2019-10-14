from typing import Dict

from envs.mechanics.pre_card import PreCard
from envs.mechanics.enums import GemColor
from envs.mechanics.gems_collection import GemsCollection


class Noble(PreCard):
    def __init__(self,
                 name: str,
                 id : int,
                 minimum_possesions: GemsCollection,
                 victory_points: int) -> None:
        """Parameters:
                _ _ _ _ _ _
                name: Name of the card (string).
                id: Identificator of the card (integer). Useful for one-hot encoding of the card.
                minimum_possesions: Dictionary with keys being gem color and values being integers. This dictionary
                describes the minimum possesions for a player to claim this noble.
                profit: Discount that this card gives when buying next cards.
                vistory_points: Victory points given by this card."""
        super().__init__(name, id, minimum_possesions, victory_points)
