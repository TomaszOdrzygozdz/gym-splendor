from typing import Dict

from envs.mechanics.card import Card
from envs.mechanics.enums import *
from envs.mechanics.noble import Noble
from envs.data.game_settings import *

import envs.mechanics.Utils.utils_functions as utils
from functools import reduce


class PlayersHand:
    """A class that describes possesions of one player."""
    def __init__(self) -> None:
        self.coins_possesed = {color : 0 for color in GemColor}
        self.cards_possesed = set()
        self.reserved_cards = set()
        self.nobles_possesed = set()

    def add_card_to_my_possesions(self,
                 card: Card) -> None:
        self.cards_possesed.add(card)

    def add_noble_to_my_possesions(self,
                  noble_card: Noble) -> None:
        self.nobles_possesed.add(noble_card)

    def earn_coins(self,
                   coins: Dict[GemColor, int]) -> None:
        self.coins = utils.add_wallets(self.coins, coins)

    def pay_coins(self,
                  coins: Dict[GemColor, int]) -> None:
        self.coins = utils.substract_wallets(self.coins, coins)

    def take_reserved_card(self,
                           card_id: int) -> Card:
        my_card = self.find_card_by_id(card_id)
        self.reserved_cards.remove(my_card)
        return my_card

    def find_card_by_id(self,
                        card_id: int) -> Card:

        found_cards = [card for card in self.reserved_cards if card_id == card.id]

        if found_cards:
            return found_cards[0]
        else:
            return None

    def add_to_reserved(self,
                        card: Card) -> None:
        self.reserved_cards.add(card)

    def get_golden_coin(self) -> None:
        self.coins[GemColor.GOLD] += 1

    def return_coin(self,
                    color: GemColor) -> None:
        self.coins[color] -= 1

    def find_discount(self) -> Dict[GemColor, int]:
        return {color: sum([card.profit == color for card in self.cards]) for color in GemColor}

    def number_of_my_points(self) -> int:
        return sum([card.win_points for card in self.cards]) + sum([noble.win_points for noble in self.nobles])

    def number_of_my_coins(self) -> int:
        return sum(self.coins.values())

    def number_of_my_cards(self) -> int:
        return len(self.cards)

    def print_my_name(self) -> None:
        print(self.name)

    def number_of_cards_can_afford(self, state):
        cards_can_afford = 0
        for card in state.board.cards_on_table:
            discount = self.find_discount()
            new_price = utils.how_much_to_pay(card.price, discount)
            if utils.can_afford_card(new_price, self.coins):
                cards_can_afford += 1

        return cards_can_afford


    def value_of_cards_can_afford(self, state):
        value_of_cards_can_afford = 0
        for card in state.board.cards_on_table:
            discount = self.find_discount()
            new_price = utils.how_much_to_pay(card.price, discount)
            if utils.can_afford_card(new_price, self.coins):
                value_of_cards_can_afford += card.win_points

        return value_of_cards_can_afford