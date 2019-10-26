from gym import Env

from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.game_settings import *
from gym_splendor_code.envs.mechanics.splendor_action_space import SplendorActionSpace
from gym_splendor_code.envs.mechanics.splendor_observation_space import SplendorObservationSpace
from gym_splendor_code.envs.mechanics.state import State
import simplejson as json

class SplendorEnv(Env):
    """ Description:
        This environment runs the game Splendor."""

    metadata = {'render.modes': ['human']}

    def __init__(self, strategies = None):

        self.current_state_of_the_game = State()
        self.action_space = SplendorActionSpace()
        self.action_space.update(self.current_state_of_the_game)
        self.observation_space = SplendorObservationSpace()
        self.is_done = False
        self.end_episode_mode = 'instant_end'
        self.gui = None

        #Create initial state of the game
    def setup_state(self, from_state = None):
        self.current_state_of_the_game.setup_state(from_state)

    def active_players_hand(self):
        return self.current_state_of_the_game.active_players_hand()

    def vectorize_state(self, output_file = None):
        state = str(self.current_state_of_the_game.vectorize()).replace("set()", "NULL")

        if output_file is not None:
            with open(output_file, 'w') as json_file:
                json.dump(state, json_file)

    def vectorize_action_space(self, output_file = None):
        state = str(self.action_space.vectorize()).replace("set()", "NULL")

        if output_file is not None:
            with open(output_file, 'w') as json_file:
                json.dump(state, json_file)

    def step(self, action: Action):
        """
        Executes action on the environment. Action is performed on the current state of the game. The are two modes for
        is_done: instant_end - the episode ends instantly when any player reaches the number of points equal POINTS_TO_WIN and
        let_all_move - when some player reaches POINTS_TO_WIN we allow all players to move (till the end of round) and then
        we end the episode.
        Reward: 1 if the action gives POINTS_TO_WIN to the player and episode is not yet ended (taking actions when episode ended
        is considered as loosing), -1 if episode ended, 0 if episode is not yet ended and the action does not give enough
        points to the player.

        :param action: action to take
        :return: observation, reward, is_done, info
        """
        """Performs one action on the current state of the game. """
        self.action_space.update(self.current_state_of_the_game)
        assert self.action_space.contains(action), '{} of type {} is not valid action'.format(action, type(action))
        action.execute(self.current_state_of_the_game)
        #We find the reward:
        reward = 0
        if not self.is_done:
            if self.current_state_of_the_game.previous_players_hand().number_of_my_points() >= POINTS_TO_WIN:
                reward = 1
        if self.is_done:
            reward = -1

        self.is_done_update(self.end_episode_mode)
        self.action_space.update(self.current_state_of_the_game)
        return self.observation_space.state_to_observation(self.current_state_of_the_game), reward, self.is_done, {}

    def is_done_update(self, end_episode_mode = 'instant_end'):
        if end_episode_mode == 'instant_end':
            if self.current_state_of_the_game.active_players_hand().number_of_my_points() >= POINTS_TO_WIN:
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

    def show_warning(self, action):
        if self.gui is not None:
            self.gui.show_warning(action)

    def show_last_action(self, action):
        if self.gui is not None:
            self.gui.show_last_action(action)

    def render(self, mode='human', interactive = False):
        """Creates window if necessary, then renders the state of the game """
        if self.gui is None:
            self.gui = SplendorGUI()

        #clear gui:
        self.gui.clear_all()
        for card in self.current_state_of_the_game.board.cards_on_board:
            self.gui.draw_state(self.current_state_of_the_game)
