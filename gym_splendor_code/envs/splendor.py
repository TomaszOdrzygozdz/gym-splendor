import time
from copy import deepcopy

from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation, StochasticObservation, \
    SplendorObservation
from gym_splendor_code.envs.mechanics.game_settings import USE_TKINTER
from gym_open_ai import Env

from gym_splendor_code.envs.data.data_loader import load_all_cards, load_all_nobles
from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.game_settings import *
from gym_splendor_code.envs.mechanics.splendor_action_space import SplendorActionSpace
from gym_splendor_code.envs.mechanics.splendor_observation_space import SplendorObservationSpace
from gym_splendor_code.envs.mechanics.state import State

from typing import List

from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict


class SplendorEnv(Env):
    """ Description:
        This environment runs the game Splendor."""

    metadata = {'render.modes': ['human']}

    def __init__(self):

        #load all cards and nobles
        self.all_cards = load_all_cards()
        self.all_nobles = load_all_nobles()

        self.current_state_of_the_game = State(all_cards=self.all_cards, all_nobles=self.all_nobles)
        self.action_space = SplendorActionSpace()
        self.update_actions()
        self.observation_space = SplendorObservationSpace(all_cards=self.all_cards, all_nobles=self.all_nobles)
        self.is_done = False
        self.first_winner = None
        self.draw = False
        self.end_episode_mode = 'instant_end'
        self.gui = None


        #Create initial state of the game

    def load_state_from_dict(self, state_as_dict: StateAsDict):
        self.current_state_of_the_game = state_as_dict.to_state()
        self.is_done = False

    def active_player_id(self):
        return self.current_state_of_the_game.active_player_id

    def active_players_hand(self):
        return self.current_state_of_the_game.active_players_hand()


    def state_to_dict(self):
        return self.current_state_of_the_game.to_dict()

    def action_space_to_dict(self):
        return self.action_space.to_dict()


    def step(self, mode, action: Action, return_observation=True, ensure_correctness = False):
        """
        Executes action on the environment. Action is performed on the current state of the game.


        The are two modes for is_done: instant_end - the episode ends instantly when any player reaches the number of
        points equal POINTS_TO_WIN and let_all_move - when some player reaches POINTS_TO_WIN we allow all players to move
        (till the end of round) and then we end the episode. Reward: 1 if the action gives POINTS_TO_WIN to the player
        and episode is not yet ended (taking actions when episode ended is considered as loosing), -1 if episode ended,
        0 if episode is not yet ended and the action does not give enough points to the player.

        :param
        action: action to take
        ensure_correctness: True if you want the enivornment check if the action is legal, False if you are sure
        :return: observation, reward, is_done, info
        """
        """Performs one action on the current current_state of the game. """
        info = {}
        if action is not None:
            if ensure_correctness:
                self.update_actions()
                assert self.action_space.contains(action), '{} is not valid action'.format(action)
            action.execute(self.current_state_of_the_game)

        # We find the reward:
        reward = 0

        if action is None:
            info = {'Warning' : 'There was no action.'}
            self.is_done = True
            self.first_winner = self.current_state_of_the_game.previous_player_id()
            reward = -1


        #if self.first_winner is not None:
        if self.current_state_of_the_game.previous_player_id() == self.first_winner:
            reward = 1
        if self.current_state_of_the_game.previous_player_id() != self.first_winner:
            reward = -1

        if self.first_winner is None:
            if not self.is_done:
                if self.current_state_of_the_game.previous_players_hand().number_of_my_points() >= POINTS_TO_WIN:
                    reward = 1
                    self.first_winner = self.current_state_of_the_game.previous_player_id()
                    self.is_done = True

        if return_observation:
            if mode == 'deterministic':
                observation_to_show = DeterministicObservation(self.current_state_of_the_game)
            if mode == 'stochastic':
                observation_to_show = StochasticObservation(self.current_state_of_the_game)
            return observation_to_show, reward, self.is_done, {'winner_id' : self.first_winner}

        if return_observation == False:
            return None, reward, self.is_done, {'winner_id' : self.first_winner}


    def is_done_update(self):
        if self.current_state_of_the_game.previous_players_hand().number_of_my_points() >= POINTS_TO_WIN:
            self.is_done = True

    def update_actions(self):
        self.action_space.update(self.current_state_of_the_game)
        self.vectorize_action_space()

    def update_actions_light(self):
        self.action_space.update(self.current_state_of_the_game)

    def current_action_space(self):
        self.update_actions()
        return self.action_space

    def show_warning(self, action):
        if self.gui is not None:
            self.gui.show_warning(action)

    def show_last_action(self, action):
        if self.gui is not None:
            self.gui.show_last_action(action)

    def load_observation(self, observation : SplendorObservation):
        self.is_done = False
        self.first_winner = None
        self.current_state_of_the_game = observation.recreate_state()

    def set_active_player(self, id: int)->None:
        self.current_state_of_the_game.active_player_id = id

    def set_players_names(self, list_of_names: List[str])->None:
        for i, name in enumerate(list_of_names):
            self.current_state_of_the_game.list_of_players_hands[i].name = name

    def points_of_player_by_id(self, id: int)-> int:
        return self.current_state_of_the_game.list_of_players_hands[id].number_of_my_points()

    def render(self, mode='human', interactive = True):
        """Creates window if necessary, then renders the current_state of the game """
        if self.gui is None:
            self.gui = SplendorGUI()

        self.gui.interactive = interactive

        #clear gui:
        #self.gui.clear_all()
        #draw current_state
        self.gui.draw_state(self.current_state_of_the_game)

    def reset(self):
        self.is_done = False
        self.first_winner = None
        self.current_state_of_the_game = State(all_cards=self.all_cards, all_nobles=self.all_nobles)
        self.update_actions()

    def show_observation(self, mode):
        self.update_actions()
        if mode == 'deterministic':
            return DeterministicObservation(self.current_state_of_the_game)
        if mode == 'stochastic':
            return StochasticObservation(self.current_state_of_the_game)

    def previous_player_id(self):
        return self.current_state_of_the_game.previous_player_id()

    def previous_players_hand(self):
        return self.current_state_of_the_game.previous_players_hand()

    def clone_state(self):
        observation = DeterministicObservation(self.current_state_of_the_game)
        return observation.recreate_state()

    def restore_state(self, state):
        self.current_state_of_the_game = state

    def vectorize_observation_space(self):
        pass

    def vectorize_action_space(self):
        pass
