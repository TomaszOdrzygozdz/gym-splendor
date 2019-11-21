from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.splendor import SplendorEnv, POINTS_TO_WIN


class SplendorFullState(SplendorEnv):

    def __init__(self):
        super().__init__()

    def full_state_step(self, action: Action):

        who_won_the_game = None

        if action is not None:
            action.execute(self.current_state_of_the_game)

        else:
            info = {'Warning': 'There was no action.'}
            self.is_done = True

        # We find the reward:
        reward = 0
        if not self.is_done:
            if self.current_state_of_the_game.previous_players_hand().number_of_my_points() >= POINTS_TO_WIN:
                reward = 1
                who_won_the_game = self.current_state_of_the_game.previous_player_id()
        if self.is_done:
            reward = -1

        self.is_done_update(self.end_episode_mode)

        return self.current_state_of_the_game, reward, self.is_done, who_won_the_game
