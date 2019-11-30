from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from gym_splendor_code.envs.data.data_loader import name_to_card_dict, name_to_noble_dict, load_all_cards, \
    load_all_nobles
from gym_splendor_code.envs.mechanics.players_hand import PlayersHand
from gym_splendor_code.envs.mechanics.state import State



class SplendorObservation:

    def __init__(self, name):
        self.name = name

    def clone(self):
        raise NotImplementedError

    def recreate_state(self):
        raise NotImplementedError


class DeterministicObservation(SplendorObservation):

    def __init__(self, state : State):
        super().__init__('deterministic')
        self.observation_dict = StateAsDict(state)

    def recreate_state(self):
        return self.observation_dict.to_state()

class StochasticObservation(SplendorObservation):

    all_cards = load_all_cards()
    all_nobles = load_all_nobles()

    def __init__(self, state : State):
        super().__init__('stochastic')
        self.observation_dict = self.state_to_stochastic_observation(state)

    def recreate_state(self):
        """Loads observation and return a current_state that agrees with the observation. Warning: this method is ambiguous,
        that is, many states can have the same observation (they may differ in the order of hidden cards)."""
        state = State(all_cards=StochasticObservation.all_cards, all_nobles=StochasticObservation.all_nobles, prepare_state=False)
        cards_on_board_names = self.observation_dict['cards_on_board_names']
        nobles_on_board_names = self.observation_dict['nobles_on_board_names']
        for card_name in cards_on_board_names:
            card = name_to_card_dict[card_name]
            state.board.cards_on_board.add(card)
            state.board.deck.decks_dict[card.row].remove(card)

        for noble_name in nobles_on_board_names:
            noble = name_to_noble_dict[noble_name]
            state.board.nobles_on_board.add(noble)
            state.board.deck.deck_of_nobles.remove(noble)

        state.board.gems_on_board = self.observation_dict['gems_on_board']

        players_hands = []
        for player_observation in self.observation_dict['players_hands']:
            players_hand = PlayersHand()
            players_hand.gems_possessed = player_observation['gems_possessed']
            for card_name in player_observation['cards_possessed_names']:
                card = name_to_card_dict[card_name]
                players_hand.cards_possessed.add(card)
                state.board.deck.decks_dict[card.row].remove(card)
            for card_name in player_observation['cards_reserved_names']:
                card = name_to_card_dict[card_name]
                players_hand.cards_reserved.add(card)
                state.board.deck.decks_dict[card.row].remove(card)
            for noble_name in player_observation['nobles_possessed_names']:
                noble = name_to_noble_dict[noble_name]
                players_hand.nobles_possessed.add(noble)
                state.board.deck.deck_of_nobles.remove(noble)
            players_hands.append(players_hand)

        state.active_player_id = self.observation_dict['active_player_id']
        state.list_of_players_hands = players_hands
        return state


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


