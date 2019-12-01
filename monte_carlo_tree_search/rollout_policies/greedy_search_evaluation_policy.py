import random
import numpy as np
import gym

from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_reservations
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_buys
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_trades
from monte_carlo_tree_search.rollout_policies.abstract_rolluot_policy import RolloutPolicy
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN



class GreedySearchEvaluationPolicy(RolloutPolicy):

    def __init__(self,
                    weight: list = [100,2,2,1,0.1],
                    decay: float = 0.9,
                    depth: int = 3,
                    breadth: int  = 2,
                    environment_id: str = 'gym_splendor_code:splendor-v0'):
        super().__init__()
        self.weight = weight
        #we create own gym-splendor enivronemt to have access to its functionality
        #We specify the name of the agent
        self.weight = weight
        self.decay = decay
        self.depth = depth
        self.breadth = breadth

        self.env = gym.make(environment_id)
        self.env_dict = {lvl : None for lvl in range(1, self.depth)}


    def choose_action(self, state : State) -> Action:
        numerator = self.depth - 1
        self.env_dict[numerator] = StateAsDict(self.env.current_state_of_the_game)
        self.restore_env()
        current_points = self.env.current_state_of_the_game.active_players_hand().number_of_my_points()

        if len(self.env.action_space.list_of_actions)>0:
            actions = []
            points = []
            for action in self.env.action_space.list_of_actions:
                ae = action.evaluate(self.env.current_state_of_the_game)
                potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                                 self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                                 self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))
                points.append(potential_reward)
                actions.append(action)

            values = set(points)
            if len(values) >= self.breadth:
                actions = [actions[i] for i, point in enumerate(points) if  point >= sorted(values)[-self.breadth]]
            if len(actions) > 1:
                actions_ = []
                points_ = []
                for action in actions:
                    ae = action.evaluate(self.env.current_state_of_the_game)
                    potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                                     self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                                     self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))
                    potential_reward -= self.decay * self.deep_evaluation(action, numerator - 1)

                    points_.append(potential_reward)
                    actions_.append(action)
                    self.restore_env(numerator)

                actions = [actions_[i] for i, point in enumerate(points_) if  point >= sorted(set(points_))[-1]]

                self.env.reset()
                self.env.load_state_from_dict(self.env_dict[numerator])

            return random.choice(actions)

        else:
            return None


    def eval_leaf(self, state : State) -> float:
        numerator = self.depth - 1
        self.env_dict[numerator] = StateAsDict(self.env.current_state_of_the_game)
        self.restore_env()
        current_points = self.env.current_state_of_the_game.active_players_hand().number_of_my_points()

        if len(self.env.action_space.list_of_actions)>0:
            actions = []
            points = []
            for action in self.env.action_space.list_of_actions:
                ae = action.evaluate(self.env.current_state_of_the_game)
                potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                                 self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                                 self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))
                points.append(potential_reward)
                actions.append(action)

            values = set(points)
            if len(values) >= self.breadth:
                actions = [actions[i] for i, point in enumerate(points) if  point >= sorted(values)[-self.breadth]]
            if len(actions) > 1:
                actions_ = []
                points_ = []
                for action in actions:
                    ae = action.evaluate(self.env.current_state_of_the_game)
                    potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                                     self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                                     self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))
                    potential_reward -= self.decay * self.deep_evaluation(action, numerator - 1)

                    points_.append(potential_reward)
                    self.restore_env(numerator)

                self.env.reset()
                self.env.load_state_from_dict(self.env_dict[numerator])

            return max(points_)

        else:
            return -100

    def deep_evaluation(self, action, numerator, mode):

        self.get_temp_env(action, numerator, mode)

        if numerator > 1:
            current_points = self.env.current_state_of_the_game.active_players_hand().number_of_my_points()
            self.env_dict[numerator] = StateAsDict(self.env.current_state_of_the_game)
            if len(self.env.action_space.list_of_actions) > 0:
                actions = []
                points = []
                potential_reward_list = []
                for action in self.env.action_space.list_of_actions:
                    ae = action.evaluate(self.env.current_state_of_the_game)
                    potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                                     self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                                     self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))

                    points.append(potential_reward)
                    actions.append(action)
                    self.restore_env(numerator)

                values = set(points)
                if len(values) >= self.breadth:
                    actions = [actions[i] for i, point in enumerate(points) if  point >= sorted(values)[-self.breadth]]
                for action in actions:
                    ae = action.evaluate(self.env.current_state_of_the_game)
                    potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                                     self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                                     self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))
                    potential_reward -= self.decay * self.deep_evaluation(action, numerator - 1, mode)
                    potential_reward_list.append(potential_reward)
                    self.restore_env(numerator)

                self.restore_env(numerator + 1)
                reward = max(potential_reward_list)
            else:
                reward = self.action_to_avoid
        else:
            reward = self.evaluate_actions()

        return reward

    def evaluate_actions(self):
        current_points = self.env.current_state_of_the_game.active_players_hand().number_of_my_points()
        if len(self.env.action_space.list_of_actions) > 0:
            potential_reward_list = []
            for action in self.env.action_space.list_of_actions:
                ae = action.evaluate(self.env.current_state_of_the_game)
                potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                                 self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                                 self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))
                potential_reward_list.append(potential_reward)
            reward = max(potential_reward_list)
        else:
            reward = self.action_to_avoid
        return reward

    def get_temp_env(self, action, numerator, mode):
        observation, reward, is_done, info = self.env.step(mode, action, ensure_correctness = False)
        self.env.load_observation(observation)
        self.env.update_actions_light()

    def restore_env(self, numerator):
        self.env.is_done = False
        self.env.current_state_of_the_game = State(all_cards=self.env.all_cards, all_nobles=self.env.all_nobles)
        self.env.load_state_from_dict(self.env_dict[numerator])
        self.env.update_actions_light()
