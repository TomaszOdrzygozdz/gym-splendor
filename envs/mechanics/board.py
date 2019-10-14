import random
from functools import reduce
from typing import Set, List

from envs.mechanics.enums import *
from envs.data.game_settings import NOBLES_ON_BOARD_INITIAL, MAX_CARDS_IN_A_ROW_ON_BOARD
from envs.mechanics.noble import Noble
from envs.mechanics.card import Card
from envs.mechanics.deck import Deck
from envs.mechanics.gems_collection import GemsCollecion

class Board:

    def __init__(self,
                 all_cards: List[Card],
                 all_nobles: List[Noble],
                 gems_on_board: GemsCollecion = None) -> None:
        """Creates a board and prepares the game. This method: creates the deck of cards and the deck od nobles. We do
        not shuffle deck and do not put cards and nobles on the board here.
        Parameters:
        _ _ _ _ _ _
        all_cards: A list of cards that will be added to the deck.
        all_nobles: A list of nobles that will be added to the deck.
        gems_on_board: A collection of gems that will be placed on the board at the beginning of the game."""
        self.deck = Deck(all_cards, all_nobles)
        if gems_on_board is None:
            self.gems_on_board = GemsCollecion()
        else:
            self.gems_on_board = gems_on_board
        self.cards_on_table = set()
        self.nobles = all_nobles
        self.nobles_on_board = set()

    def shuffle(self) -> None:
        """Shuffles both: deck of cards and list of nobles."""
        self.deck.shuffle()
        random.shuffle(self.nobles)

    def lay_cards_on_table(self) -> None:
        """Puts appropriate number of cards on the board. """
        drawn_cards = map(lambda x: self.deck.pop_many(x, MAX_CARDS_IN_A_ROW_ON_BOARD), self.deck.decks_dict.keys())
        self.cards_on_table = set(reduce(lambda x, y: x + y, drawn_cards))

    def lay_nobles_on_board(self) -> None:
        """This method puts three nobles on the board."""
        self.nobles_on_board = set(self.nobles[0:NOBLES_ON_BOARD_INITIAL])