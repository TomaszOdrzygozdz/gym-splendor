from gym_splendor_code.envs.mechanics.enums import Row, GemColor
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.pre_card import PreCard


class Card(PreCard):
    def __init__(self,
                 name: str,
                 id: int,
                 row: Row,
                 price: GemsCollection,
                 discount_profit: GemColor,
                 victory_points: int) -> None:
        """Parameters:
        _ _ _ _ _ _
        name: Name of the card (string).
        id: Identificator of the card (integer). Useful for one-hot encoding of the card.
        row: Row to which this card belong (Cheap, Medium or Expensive).
        price: Dictionary with keys being gem colors and values being integers. This dictionary describes the price of
        card.
        profit: Discount that this card gives when buying next cards.
        vistory_points: Victory points given by this card."""
        super().__init__(name, id, price, victory_points)
        self.row = row
        self.discount_profit = discount_profit

    def __repr__(self):
        return 'Card(name: {}, id: {}, row: {}, price: {}, profit: {}, victory_points: {})'.format(self.name, self.id,
                                                                                                   self.row, self.price,
                                                                                                   self.discount_profit,
                                                                                                   self.victory_points)