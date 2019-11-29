from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict


class SplendorObservation:

    def __init__(self, name):
        self.name = name


class DeterministicObservation(SplendorObservation):

    def __init__(self, state : State):
        super().__init__('deterministic')
        self.state_as_dict = StateAsDict(state)

class StochasticObservation(SplendorObservation):

    def __init__(self, state : State):
        super().__init__('stochastic')
        self.observation_as_dict = StateAsDict(state)

    def state_to_stochastic_observation(self, state:State):
        cards_on_board_names = {card.name for card in state.board.cards_on_board if card is not None}
        nobles_on_board_names = {noble.name for noble in state.board.nobles_on_board if noble is not None}
        gems_on_board = state.board.gems_on_board.__copy__()
        active_player_id = state.active_player_id
        players_hands = [{'cards_possessed_names': {card.name for card in players_hand.cards_possessed if card is not None},
                          'cards_reserved_names' : {card.name for card in players_hand.cards_reserved if card is not None},
                          'nobles_possessed_names' : {noble.name for noble in players_hand.nobles_possessed if noble is not None},
                          'gems_possessed' : players_hand.gems_possessed.__copy__()} for players_hand in state.list_of_players_hands]

        return {'cards_on_board_names' : cards_on_board_names, 'nobles_on_board_names': nobles_on_board_names,
                'gems_on_board' : gems_on_board, 'active_player_id': active_player_id, 'players_hands' : players_hands}


