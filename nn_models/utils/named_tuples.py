from collections import namedtuple

from gym_splendor_code.envs.mechanics.enums import GemColor

GemsTuple = namedtuple('gems_collection', ' '.join([str(x).replace('GemColor.', '') for x in GemColor]))
PriceTuple = namedtuple('price', ' '.join([str(x).replace('GemColor.', '') for x in GemColor if x != GemColor.GOLD]))
CardTuple = namedtuple('card', 'profit price victory_points')
BoardTuple = namedtuple('board', 'gems cards nobles')
PlayerTuple = namedtuple('player', 'discount gems reserved_cards points nobles')
ObservationTuple = namedtuple('observation', 'active_player previous_player board')

GemInputTuple = namedtuple('gems_input', ' '.join([str(x).replace('GemColor.', '') for x in GemColor]))
GemEmbeddingTuple = namedtuple('gems_embeddings', ' '.join([str(x).replace('GemColor.', '') for x in GemColor]))