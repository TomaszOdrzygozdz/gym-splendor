from typing import Dict

from envs.mechanics.card import Card
from envs.mechanics.noble import Noble
from envs.mechanics.gems_collection import GemsCollection
from envs.data.game_settings import *


class PlayersHand:
    """A class that describes possessions of one player."""
    def __init__(self,
                 name : str = "Player") -> None:
        """Creates a hand with empty gems collections, empty set of cards, nobles and reserved cards.
        Parameters:
        _ _ _ _ _ _
        name: The name of player who has this hand (optional)."""
        self.name = name
        self.gems_possessed = GemsCollection()
        self.cards_possessed = set()
        self.reserved_cards = set()
        self.nobles_possessed = set()
        self.discount = GemsCollection()

    def add_card_to_my_possesions(self,
                 card: Card) -> None:
        self.cards_possessed.add(card)

    def add_noble_to_my_possesions(self,
                  noble_card: Noble) -> None:
        self.nobles_possessed.add(noble_card)

    def add_card_to_reserved(self,
                        card: Card) -> None:
        self.reserved_cards.add(card)

    def remove_reserved_card(self,
                           card: Card) -> Card:
        self.reserved_cards.remove(card)

    def find_discount(self) -> Dict[GemColor, int]:
        return {color: sum([card.profit == color for card in self.cards]) for color in GemColor}

    def number_of_my_points(self) -> int:
        return sum([card.win_points for card in self.cards]) + sum([noble.win_points for noble in self.nobles])

    def number_of_my_gems(self) -> int:
        """Returns number of gems possesed by this hand."""
        return sum(self.coins.values())

    def number_of_my_cards(self) -> int:
        """Returns number of cards possesed by this hand."""
        return len(self.cards)