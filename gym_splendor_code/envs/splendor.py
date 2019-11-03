import time
from copy import deepcopy
from gym_splendor_code.envs.mechanics.game_settings import USE_TKINTER
from gym import Env

from gym_splendor_code.envs.data.data_loader import load_all_cards, load_all_nobles
from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.game_settings import *
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from gym_splendor_code.envs.mechanics.splendor_action_space import SplendorActionSpace
from gym_splendor_code.envs.mechanics.splendor_observation_space import SplendorObservationSpace
from gym_splendor_code.envs.mechanics.state import State
import simplejson as json

from typing import List

class SplendorEnv(Env):
    """ Description:
        This environment runs the game Splendor."""

    metadata = {'render.modes': ['human']}

    def __init__(self, thread_str=''):

        #load all cards and nobles
        self.all_cards = load_all_cards()
        self.all_nobles = load_all_nobles()

        self.current_state_of_the_game = State(all_cards=self.all_cards, all_nobles=self.all_nobles)
        self.current_state_of_the_game.setup_state()
        self.action_space = SplendorActionSpace()
        self.action_space.update(self.current_state_of_the_game)
        self.observation_space = SplendorObservationSpace(all_cards=self.all_cards, all_nobles=self.all_nobles)
        self.is_done = False
        self.end_episode_mode = 'instant_end'
        self.gui = None

        #Create initial state of the game
    def setup_state(self, from_state = None, file = False, ordered_deck = False):
        self.current_state_of_the_game.setup_state(from_state, file, ordered_deck)

    def active_players_hand(self):
        return self.current_state_of_the_game.active_players_hand()

    def vectorize_state(self, output_file = None, return_var = False):
        state = str(self.current_state_of_the_game.vectorize()).replace("set()", "NULL")

        if output_file is not None:
            with open(output_file, 'w') as json_file:
                json.dump(state, json_file)

        if return_var:
            return state

    def vectorize_action_space(self, output_file = None):
        state = str(self.action_space.vectorize()).replace("set()", "NULL")

        if output_file is not None:
            with open(output_file, 'w') as json_file:
                json.dump(state, json_file)

    def step(self, action: Action, ensure_correctness = False):
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
        """Performs one action on the current state of the game. """
        info = {}
        if action is not None:
            if ensure_correctness:
                self.action_space.update(self.current_state_of_the_game)
                assert self.action_space.contains(action), '{} of type {} is not valid action'.format(action, type(action))
            action.execute(self.current_state_of_the_game)

        else:
            info = {'Warning' : 'There was no action.'}
        # We find the reward:
        reward = 0
        if not self.is_done:
            if self.current_state_of_the_game.previous_players_hand().number_of_my_points() >= POINTS_TO_WIN:
                reward = 1
        if self.is_done:
            reward = -1

        self.is_done_update(self.end_episode_mode)

        return self.observation_space.state_to_observation(self.current_state_of_the_game), reward, self.is_done, {}

    def is_done_update(self, end_episode_mode = 'instant_end'):
        if end_episode_mode == 'instant_end':
            if self.current_state_of_the_game.previous_players_hand().number_of_my_points() >= POINTS_TO_WIN:
                self.is_done = True
        if end_episode_mode == 'let_all_move':
            #the episone can end only if some has reached enough points and last player has moved
            if self.current_state_of_the_game.active_player_id == \
                    len(self.current_state_of_the_game.list_of_players_hands) - 1:
                #check if someone has reached enough points to win
                for player_hand in self.current_state_of_the_game.list_of_players_hands:
                    if player_hand.number_of_my_points() >= POINTS_TO_WIN:
                        self.is_done = True
                        break

    def update_actions(self):
        self.action_space.update(self.current_state_of_the_game)

    def current_action_space(self):
        self.action_space.update(self.current_state_of_the_game)
        return self.action_space

    def show_warning(self, action):
        if self.gui is not None:
            self.gui.show_warning(action)

    def show_last_action(self, action):
        if self.gui is not None:
            self.gui.show_last_action(action)

    def load_observation(self, observation):
        self.current_state_of_the_game = self.observation_space.observation_to_state(observation)

    def set_active_player(self, id: int)->None:
        self.current_state_of_the_game.active_player_id = id

    def set_players_names(self, list_of_names: List[str])->None:
        for i, name in enumerate(list_of_names):
            self.current_state_of_the_game.list_of_players_hands[i].name = name

    def points_of_player_by_id(self, id: int)-> int:
        return self.current_state_of_the_game.list_of_players_hands[id].number_of_my_points()

    def render(self, mode='human', interactive = True):
        """Creates window if necessary, then renders the state of the game """
        if self.gui is None:
            self.gui = SplendorGUI()

        self.gui.interactive = interactive

        #clear gui:
        self.gui.clear_all()
        #draw state
        self.gui.draw_state(self.current_state_of_the_game)

    def reset(self):
        self.is_done = False
        self.current_state_of_the_game = State(all_cards=self.all_cards, all_nobles=self.all_nobles)
        self.current_state_of_the_game.setup_state()
        self.update_actions()

    def show_observation(self):
        self.update_actions()
        return self.observation_space.state_to_observation(self.current_state_of_the_game)
