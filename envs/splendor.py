from gym import Env
from gym.utils import seeding

from envs.mechanics.action import Action
from envs.mechanics.game_settings import *
from envs.mechanics.splendor_action_space import SplendorActionSpace
from envs.mechanics.splendor_observation_space import SplendorObservationSpace
from envs.mechanics.state import State


class SplendorEnv(Env):
    """ Description:
        This environment runs the game Splendor."""

    def __init__(self, strategies = None):

        self.current_state_of_the_game = State()
        self.action_space = SplendorActionSpace()
        self.action_space.update(self.current_state_of_the_game)
        self.observation_space = SplendorObservationSpace()
        self.is_done = False
        self.end_episode_mode = 'instant_end'

        #Create initial state of the game


    def step(self, action: Action):
        """Performs one action on the current state of the game. """
        assert self.action_space.contains(action), '{} of type {} is not valid action'.format_map(action, type(action))
        action.execute(self.current_state_of_the_game)
        #First we find the reward:
        reward = 0
        if not self.is_done:
            if self.current_state_of_the_game.previous_players_hans().number_of_my_points() >= POINTS_TO_WIN:
                reward = 1
        if self.is_done:
            reward = -1

        self.is_done_update(self.end_episode_mode)
        return self.current_state_of_the_game, reward, self.is_done, {}



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

    def render(self, mode='human'):
        pass



