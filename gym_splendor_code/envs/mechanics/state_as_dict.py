from typing import Dict

from gym_splendor_code.envs.mechanics.state import State


class StateAsDict:

    def __init__(self, state: State = None):
        if state is not None:
            self.state_as_dict = state.to_dict()
        else:
            self.state_as_dict = {}

    def load_from_dict(self, dict):
        self.state_as_dict = dict

    def __getitem__(self, item):
        return self.state_as_dict[item]

    def clone(self):
        return self.state_as_dict.copy()

    def __repr__(self):
        return self.state_as_dict.__repr__()

    def to_state(self, order_deck = True):
        state = State(prepare_state=False)

        state.active_player_id = self.state_as_dict['active_player_id']
        state.list_of_players_hands[state.active_player_id].from_json(self.state_as_dict['active_player_hand'])
        state.list_of_players_hands[(state.active_player_id - 1) % len(state.list_of_players_hands)].from_json(
            self.state_as_dict['other_player_hand'])
        state.board.from_json(self.state_as_dict)

        # Adding nobles
        for i in self.state_as_dict['active_player_hand']['noble_possessed_ids']:
            state.list_of_players_hands[state.active_player_id].nobles_possessed.add(
                state.board.deck.pop_noble_by_id(i))

        for i in self.state_as_dict['other_player_hand']['noble_possessed_ids']:
            state.list_of_players_hands[
                (state.active_player_id - 1) % len(state.list_of_players_hands)].nobles_possessed.add(
                state.board.deck.pop_noble_by_id(i))

        for i in self.state_as_dict['board']['nobles_on_board']:
            state.board.nobles_on_board.add(state.board.deck.pop_noble_by_id(i))

        # Adding cards
        for i in self.state_as_dict['active_player_hand']['cards_possessed_ids']:
            state.list_of_players_hands[state.active_player_id].cards_possessed.add(state.board.deck.pop_card_by_id(i))

        for i in self.state_as_dict['active_player_hand']['cards_reserved_ids']:
            state.list_of_players_hands[state.active_player_id].cards_reserved.add(state.board.deck.pop_card_by_id(i))

        for i in self.state_as_dict['other_player_hand']['cards_possessed_ids']:
            state.list_of_players_hands[
                (state.active_player_id - 1) % len(state.list_of_players_hands)].cards_possessed.add(
                state.board.deck.pop_card_by_id(i))

        for i in self.state_as_dict['other_player_hand']['cards_reserved_ids']:
            state.list_of_players_hands[
                (state.active_player_id - 1) % len(state.list_of_players_hands)].cards_reserved.add(
                state.board.deck.pop_card_by_id(i))

        if order_deck:
            state.board.deck.order_deck(self.state_as_dict)
        else:
            state.board.deck.shuffle()

        return state

    def __repr__(self):
        return self.state_as_dict.__repr__()


