import random
import numpy as np
from agents.abstract_agent import Agent
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN


class GreedyAgentBoost(Agent):

    def __init__(self,
                name: str = "Greedy",
                weight: list = [100,2,2,1,0.1]):


        super().__init__()

        #we create own gm_open_ai-splendor enivronemt to have access to its functionality
        #We specify the name of the agent
        self.name = name + ' ' + str(weight) + ' ' + str(Agent.agents_created)
        self.weight = weight
        self.normalize_weight()

    def choose_act(self, mode) -> Action:

        #first we load observation to the private environment
        current_points = self.env.current_state_of_the_game.active_players_hand().number_of_my_points()

        if len(self.env.action_space.list_of_actions):
            actions = []
            points = []
            potential_reward_max = -20
            for action in self.env.action_space.list_of_actions:
                ae = action.evaluate(self.env.current_state_of_the_game)
                potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] +\
                                    self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] +\
                                     self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))

                actions.append(action)
                points.append(potential_reward)
            actions = [actions[i] for i, point in enumerate(points) if  point >= sorted(set(points))[-1]]
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
