import random
from functools import reduce
from typing import Set

from gym_splendor_code.envs.mechanics.game_settings import NOBLES_ON_BOARD_INITIAL, MAX_CARDS_IN_A_ROW_ON_BOARD
from gym_splendor_code.envs.mechanics.noble import Noble
from gym_splendor_code.envs.mechanics.card import Card
from gym_splendor_code.envs.mechanics.deck import Deck
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection

class Board:

    def __init__(self,
                 all_cards: Set[Card],
                 all_nobles: Set[Noble],
                 gems_on_board: GemsCollection = GemsCollection()) -> None:
        """Creates a board and prepares the game. This method: creates the deck of cards and the deck od nobles. We do
        not shuffle deck and do not put cards and nobles on the board here.
        Parameters:
        _ _ _ _ _ _
        all_cards: A list of cards that will be added to the deck.
        all_nobles: A list of nobles that will be added to the deck.
        gems_on_board: A collection of gems that will be placed on the board at the beginning of the game."""
        self.deck = Deck(all_cards, all_nobles)
        self.gems_on_board = gems_on_board
        self.cards_on_board = set()
        self.nobles_on_board = set()

    def shuffle(self) -> None:
        """Shuffles both: deck of cards and list of nobles."""
        self.deck.shuffle()

    def lay_cards_on_board(self) -> None:
        """Puts appropriate number of cards on the board. """
        drawn_cards = map(lambda x: self.deck.pop_many_from_one_row(x, MAX_CARDS_IN_A_ROW_ON_BOARD), self.deck.decks_dict.keys())
        self.cards_on_board = set(reduce(lambda x, y: x + y, drawn_cards))

    def lay_nobles_on_board(self) -> None:
        """This method puts three nobles on the board."""
        self.nobles_on_board = set(self.deck.pop_many_nobles(NOBLES_ON_BOARD_INITIAL))

    def remove_card_from_board_and_refill(self, card: Card) -> None:
        """This method removes a card from board and puts a new one (if there is non-empty deck to take from"""
        self.cards_on_board.remove(card)
        self.deck.pop_card(card.row)