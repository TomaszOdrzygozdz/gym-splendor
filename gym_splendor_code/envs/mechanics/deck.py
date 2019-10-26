import random
from typing import Set, List

from gym_splendor_code.envs.mechanics.noble import Noble
from gym_splendor_code.envs.mechanics.enums import Row
from gym_splendor_code.envs.mechanics.card import Card

class Deck:
    """This class is used to store three decks of cards used in the game: cheap deck, medium deck and expensive deck."""
    def __init__(self,
                 set_of_cards: Set[Card],
                 set_of_nobles: Set[Noble]) -> None:

        self.decks_dict = {row : [] for row in Row}

        #Here we put every card from the list_of_card to an appropriate deck:
        for card in set_of_cards:
            self.decks_dict[card.row].append(card)

        self.deck_of_nobles = list(set_of_nobles)

    def pop_card(self,
            row: Row) -> Card:
        """Pops a card from a given row. Returns this card and removes it from the deck."""
        if len(self.decks_dict[row]) > 0:
            return self.decks_dict[row].pop(0)

    def pop_many_from_one_row(self,
                 row: Row,
                 number: int = 4) -> List[Card]:
        """Pops many cards from a given row."""
        return [self.pop_card(row) for _ in range(number)]

    def pop_many_nobles(self,
                 number: int = 3) -> List[Noble]:
        """Pops many cards from a given row."""
        return [self.deck_of_nobles.pop() for _ in range(number)]


    def pop_many_nobles(self,
                        number: int = 3) -> List[Noble]:
        """Pops many nobles from the deck."""
        return self.deck_of_nobles[0:number]

    def shuffle(self) -> None:
        """Shuffles both deck of cards and deck of nobles."""
        for list_of_cards in self.decks_dict.values():
            random.shuffle(list_of_cards)

        random.shuffle(self.deck_of_nobles)

    def how_many_cards_left(self, row: Row) -> int:
        """Returns number of unrevealed card in a given row."""
        return len(self.decks_dict[row])

    def pop_card_by_id(self,
                        id: int):
        for row in Row:
            ids = [i for i,x in enumerate([x.vectorize() for x in self.decks_dict[row]]) if x == id]
            if ids:
                return self.decks_dict[row].pop(ids[0])

    def pop_cards_from_id_list(self,
                        list: list,
                        row: Row):
        cards = []
        for id in list:
            ids = [i for i, x in enumerate([x.vectorize() for x in self.decks_dict[row]]) if x == id]
            if ids:
                cards.append(self.decks_dict[row].pop(ids[0]))
        return cards

    def pop_noble_by_id(self,
                        id: int):
        ids = [i for i,x in enumerate([x.vectorize() for x in self.deck_of_nobles]) if x == id]
        if ids:
            return self.deck_of_nobles.pop(ids[0])
