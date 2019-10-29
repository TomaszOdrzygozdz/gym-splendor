import random
import numpy as np
from agent import Agent
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN


class RandomAgent(Agent):

    def __init__(self, distribution='uniform_on_types'):
        """:param:
        distribution: argument determining how action is chosen at random. Possible options are:
        uniform - this draws from random distribution on all legal action
        uniform_on_types - first we draw a type of action at random (with uniform distribution on existing types) and
        later choose at random an action of this type from uniform distribution along actions of this type
        first_buy - if it is possible to buy a card we choose buying action at ranodm with uniform distribution, if not
        we choose action at random."""

        super().__init__()

        self.distribution = distribution
        #we create own gym-splendor enivronemt to have access to its functionality
        #We specify the name of the agent
        self.name = 'RandomAgent - ' + self.distribution

    def choose_action(self, observation) -> Action:

        #first we load observation to the private environment
        self.env.load_observation(observation)
        self.env.update_actions()

        if self.distribution == 'uniform':
            return random.choice(self.env.action_space.list_of_actions)
        if self.distribution == 'uniform_on_types':
            chosen_action_type = random.choice([action_type for action_type in
                                                self.env.action_space.actions_by_type.keys() if
                                                len(self.env.action_space.actions_by_type[action_type]) > 0])
            return random.choice(self.env.action_space.actions_by_type[chosen_action_type])
        if self.distribution == 'first_buy':
            if len(self.env.action_space.actions_by_type['buy']) > 0:
                return random.choice(self.env.action_space.actions_by_type['buy'])
            else:
                return random.choice(self.env.action_space.list_of_actions)


class GreedyAgent(Agent):

    def __init__(self,
                name: str = "Greedy",
                weight: float = 0.5):
        """:param:
        distribution: argument determining how action is chosen at random. Possible options are:
        uniform - this draws from random distribution on all legal action
        uniform_on_types - first we draw a type of action at random (with uniform distribution on existing types) and
        later choose at random an action of this type from uniform distribution along actions of this type
        first_buy - if it is possible to buy a card we choose buying action at ranodm with uniform distribution, if not
        we choose action at random."""

        super().__init__()

        #we create own gym-splendor enivronemt to have access to its functionality
        #We specify the name of the agent
        self.name = name + str(weight)
        self.weight = weight

    def choose_action(self, observation) -> Action:

        #first we load observation to the private environment
        self.env.load_observation(observation)
        self.env.update_actions()
        current_points = self.env.current_state_of_the_game.active_players_hand().number_of_my_points()

        if len(self.env.action_space.list_of_actions):
            actions = []
            potential_reward_max = -20
            for action in self.env.action_space.list_of_actions:
                ae = action.evaluate(self.env.current_state_of_the_game)
                potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN)* 100 + 2 * ae["card"][2] + 2 *ae["nobles"] + ae["card"][0] + self.weight * sum(ae["gems_flow"]))
                if potential_reward > potential_reward_max:
                    potential_reward_max = potential_reward
                    actions = []
                    actions.append(action)
                elif potential_reward == potential_reward_max:
                    actions.append(action)

            return random.choice(actions)

        else:
            return None

class GreedyAgentBoost(Agent):

    def __init__(self,
                name: str = "Greedy",
                weight: list = []):
        """:param:
        distribution: argument determining how action is chosen at random. Possible options are:
        uniform - this draws from random distribution on all legal action
        uniform_on_types - first we draw a type of action at random (with uniform distribution on existing types) and
        later choose at random an action of this type from uniform distribution along actions of this type
        first_buy - if it is possible to buy a card we choose buying action at ranodm with uniform distribution, if not
        we choose action at random."""

        super().__init__()

        #we create own gym-splendor enivronemt to have access to its functionality
        #We specify the name of the agent
        self.name = name
        self.weight = weight
        self.normalize_weight()

    def choose_action(self, observation) -> Action:

        #first we load observation to the private environment
        self.env.load_observation(observation)
        self.env.update_actions()
        current_points = self.env.current_state_of_the_game.active_players_hand().number_of_my_points()

        if len(self.env.action_space.list_of_actions):
            actions = []
            potential_reward_max = -20
            for action in self.env.action_space.list_of_actions:
                ae = action.evaluate(self.env.current_state_of_the_game)
                potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN) * self.weight[0] + self.weight[1] * ae["card"][2] + self.weight[2] *ae["nobles"] + self.weight[3] * ae["card"][0] + self.weight[4] * sum(ae["gems_flow"]))
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
        lr = lr * (ratio/0.5 - 1)
        if lr < 0:
            lr = -(1 + lr) * 0.02
        self.weight = [a + b *lr for a, b in zip(self.weight, list)]
        self.normalize_weight()
