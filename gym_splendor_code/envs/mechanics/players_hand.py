from gym_splendor_code.envs.mechanics.card import Card
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.enums import GemColor
from gym_splendor_code.envs.mechanics.game_settings import *


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
        self.cards_reserved = set()
        self.nobles_possessed = set()
        self.stored_discount = None

    def discount(self):
        """Returns gems collection that contains the sum of profits of card possessed by the players_hand."""
        discount_dict = {gem_color : 0 for gem_color in GemColor}
        for card in self.cards_possessed:
            discount_dict[card.discount_profit] += 1
        return GemsCollection(discount_dict)

    def can_afford_card(self,
                        card: Card,
                        discount: GemsCollection = None) -> bool:
        """Returns true if players_hand can afford card"""
        if discount is None:
            discount = self.discount()
        price_after_discount = card.price % discount
        missing_gems = 0
        for gem_color in GemColor:
            if gem_color != GemColor.GOLD:
                missing_gems += max(price_after_discount.value(gem_color) - self.gems_possessed.value(gem_color),0)
        return self.gems_possessed.value(GemColor.GOLD) >= missing_gems

    def min_gold_needed_to_buy_card(self,
                                    card: Card)->int:
        price_after_discount = card.price % self.discount()
        missing_gems = 0
        for gem_color in GemColor:
            if gem_color != GemColor.GOLD:
                missing_gems += max(price_after_discount.value(gem_color) - self.gems_possessed.value(gem_color),0)
        return missing_gems


    def can_reserve_card(self):
        return len(self.cards_reserved) < MAX_RESERVED_CARDS

    def number_of_my_points(self) -> int:
        return sum([card.victory_points for card in self.cards_possessed]) + sum([noble.victory_points for noble in self.nobles_possessed])

    def vectorize(self):
        return {'noble_possessed_ids' : {x.vectorize() for x in self.nobles_possessed},
                'cards_possessed_ids' : {x.vectorize() for x in self.cards_possessed},
                'cards_reserved_ids' : {x.vectorize() for x in self.cards_reserved},
                'gems_possessed' : self.gems_possessed.vectorize(),
                'name': self.name}

    def from_vector(self, vector):
        self.name  = vector['name']
        gems = vector['gems_possessed']
        self.gems_possessed = self.gems_possessed +  GemsCollection({GemColor.GOLD: gems[0], GemColor.RED: gems[1],
                                    GemColor.GREEN: gems[2], GemColor.BLUE: gems[3],
                                    GemColor.WHITE: gems[4], GemColor.BLACK: gems[5]})

        for i in vector['noble_possessed_ids']:
            self.nobles_possessed.add(noble[i])
