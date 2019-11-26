from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from gym_splendor_code.envs.splendor import SplendorEnv, POINTS_TO_WIN


class SplendorDeterministic(SplendorEnv):

    def __init__(self):
        super().__init__()

    def deterministic_step(self, action: Action):

        reward = 0
        who_won_the_game = None
        if action is not None:
            action.execute(self.current_state_of_the_game)
        else:
            info = {'Warning': 'There was no action.'}
            self.is_done = True
            reward = -1

        # We find the reward:

        if not self.is_done:
            if self.current_state_of_the_game.previous_players_hand().number_of_my_points() >= POINTS_TO_WIN:
                reward = 1
                who_won_the_game = self.current_state_of_the_game.previous_player_id()
        if self.is_done:
            reward = -1

        self.is_done_update(self.end_episode_mode)

        return self.current_state_of_the_game, reward, self.is_done, who_won_the_game
