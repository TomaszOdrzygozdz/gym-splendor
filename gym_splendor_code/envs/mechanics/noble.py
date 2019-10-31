from gym_splendor_code.envs.mechanics.pre_card import PreCard
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection

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

    def __eq__(self, other):
        #safer method of noble comparison:
        # condition1 = self.name == other.name
        # condition2 = self.price == other.price
        # return condition1 and condition2 and condition2
        #way faster comparison:
        return self.name == other.name

    def __hash__(self):
        return self.id