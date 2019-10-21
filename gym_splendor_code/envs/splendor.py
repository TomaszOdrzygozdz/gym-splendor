from gym import Env

from gym_splendor_code.envs.graphics.splendor_gui import SplendorGUI
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.game_settings import *
from gym_splendor_code.envs.mechanics.splendor_action_space import SplendorActionSpace
from gym_splendor_code.envs.mechanics.splendor_observation_space import SplendorObservationSpace
from gym_splendor_code.envs.mechanics.state import State


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
        assert self.action_space.contains(action), '{} of type {} is not valid action'.format_map(action, type(action))
        action.execute(self.current_state_of_the_game)
        #We find the reward:
        reward = 0
        if not self.is_done:
            if self.current_state_of_the_game.previous_players_hans().number_of_my_points() >= POINTS_TO_WIN:
                reward = 1
        if self.is_done:
            reward = -1

        self.is_done_update(self.end_episode_mode)
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

    def render(self, mode='human', interactive = False):
        """Creates window if necessary, then renders the state of the game """
        if self.gui is None:
            self.gui = SplendorGUI()

        self.gui.draw_state(self.current_state_of_the_game)
        self.gui.keep_window_open()



