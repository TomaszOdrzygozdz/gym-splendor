import csv
from typing import Set

from gym_splendor_code.envs.data.files_paths import CARDS_DATABASE_FILE, NOBLES_DATABASE_FILE
from gym_splendor_code.envs.mechanics.card import Card
from gym_splendor_code.envs.mechanics.enums import Row, GemColor
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.noble import Noble

# dictionaries useful to convert vector of state to state:
id_to_card_dict = {}
id_to_noble_dict = {}
name_to_card_dict = {}
name_to_noble_dict ={}

# string to row dictionary
str_to_row = {'Row.CHEAP': Row.CHEAP,
              'Row.MEDIUM': Row.MEDIUM,
              'Row.EXPENSIVE': Row.EXPENSIVE}

# str to gem color dictionary:
str_to_color = {'GemColor.RED': GemColor.RED,
                'GemColor.GREEN': GemColor.GREEN,
                'GemColor.BLUE': GemColor.BLUE,
                'GemColor.WHITE': GemColor.WHITE,
                'GemColor.BLACK': GemColor.BLACK,
                'GemColor.GOLD': GemColor.GOLD}


def load_all_cards(file: str = CARDS_DATABASE_FILE) -> Set[Card]:
    """Loads information about cards from file and returns a set of cards."""
    set_of_cards = set()
    cards_database_file = open(file)
    reader = csv.reader(cards_database_file)
    _ = next(reader, None)
    card_id = 0
    for row in reader:
        price = GemsCollection({GemColor.BLACK: int(row[2]), GemColor.WHITE: int(row[3]), GemColor.RED: int(row[4]),
                                GemColor.BLUE: int(row[5]), GemColor.GREEN: int(row[6]), GemColor.GOLD: 0})
        card = Card(row[0], card_id, str_to_row[row[1]], price, str_to_color[row[7]], int(row[8]))
        set_of_cards.add(card)
        id_to_card_dict[card_id] = card
        name_to_card_dict[card.name] = card
        card_id += 1
    return set_of_cards


def load_all_nobles(file: str = NOBLES_DATABASE_FILE) -> Set[Noble]:
    """Loads information about nobles from file and returns a set of cards."""
    set_of_nobles = set()
    nobles_database_file = open(file)
    reader = csv.reader(nobles_database_file)
    _ = next(reader, None)
    noble_id = 100
    for row in reader:
        price = GemsCollection({GemColor.BLACK: int(row[1]), GemColor.WHITE: int(row[2]), GemColor.RED: int(row[3]),
                                GemColor.BLUE: int(row[4]), GemColor.GREEN: int(row[5]), GemColor.GOLD: 0})
        new_noble = Noble(row[0], noble_id, price, int(row[6]))
        set_of_nobles.add(new_noble)
        id_to_noble_dict[noble_id] = new_noble
        name_to_noble_dict[new_noble.name] = new_noble
        noble_id += 1
    return set_of_nobles


def id_to_card(id: int) -> Card:
    """Returns card by its id."""
    return id_to_card_dict[id]


def id_to_noble(id: int) -> Card:
    """Returns noble by its id."""
    return id_to_noble_dict[id]
