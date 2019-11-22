import random
import numpy as np
from agent import Agent
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN


class GreedyAgentBoost(Agent):

    def __init__(self,
                name: str = "Greedy",
                weight: list = []):


        super().__init__()

        #we create own gym-splendor enivronemt to have access to its functionality
        #We specify the name of the agent
        self.name = name + ' ' + str(weight) + ' ' + str(Agent.agents_created)
        self.weight = weight
        self.normalize_weight()

    def choose_action(self, observation, previous_actions) -> Action:

        #first we load observation to the private environment
        self.env.load_observation_light(observation)
        self.env.update_actions_light()
        current_points = self.env.current_state_of_the_game.active_players_hand().number_of_my_points()

        if len(self.env.action_space.list_of_actions):
            actions = []
            potential_reward_max = -20
            for action in self.env.action_space.list_of_actions:
                ae = action.evaluate(self.env.current_state_of_the_game)
                potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                    self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                     self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))
                if potential_reward > potential_reward_max:
                    potential_reward_max = potential_reward
                    actions = []
                    actions.append(action)
                elif potential_reward == potential_reward_max:
                    actions.append(action)

            return random.choice(actions)

        else:
            return None

    def normalize_weight(self):
        if np.linalg.norm(self.weight) > 0:
            self.weight = self.weight/np.linalg.norm(self.weight)

    def update_weight(self, list, lr, ratio):
        list = list/np.linalg.norm(list)
        lr = lr * ratio
        self.weight = [a + b *lr for a, b in zip(self.weight, list)]
        self.normalize_weight()
