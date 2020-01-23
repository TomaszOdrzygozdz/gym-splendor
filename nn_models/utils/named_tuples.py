from collections import namedtuple

from gym_splendor_code.envs.mechanics.enums import GemColor
from gym_splendor_code.envs.mechanics.game_settings import MAX_CARDS_ON_BORD

def tuple_to_str(fields, prefix):
    list_of_fields = [prefix + x for x in fields]
    return ' '.join(list_of_fields) + ' '

GemsTuple = namedtuple('gems_collection', ' '.join([str(x).replace('GemColor.', '') for x in GemColor]))
PriceTuple = namedtuple('price', ' '.join([str(x).replace('GemColor.', '') for x in GemColor if x != GemColor.GOLD]))
CardTuple = namedtuple('card', 'profit ' + tuple_to_str(PriceTuple._fields, 'price_') + ' victory_points')
NobleTuple = namedtuple('noble', tuple_to_str(PriceTuple._fields, 'price_'))
BoardTuple = namedtuple('board', tuple_to_str(GemsTuple._fields, 'board_gems_') + tuple_to_str(CardTuple._fields, 'cards_') +
                        tuple_to_str(NobleTuple._fields, 'nobles_')+' cards_mask nobles_mask')
PlayerTuple = namedtuple('player',  tuple_to_str(GemsTuple._fields, 'player_gems_')
                         + tuple_to_str(CardTuple._fields, 'res_cards_') + ' points nobles')


# PlayerTuple = namedtuple('player', 'discount gems reserved_cards points nobles')
# ObservationTuple = namedtuple('observation', 'active_player previous_player board')
#
# GemInputTuple = namedtuple('gems_input', ' '.join([str(x).replace('GemColor.', '') for x in GemColor]))
# GemEmbeddingTuple = namedtuple('gems_embeddings', ' '.join([str(x).replace('GemColor.', '') for x in GemColor]))
