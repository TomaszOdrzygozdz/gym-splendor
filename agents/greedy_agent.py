import random
import numpy as np
from agent import Agent
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN


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

        if len(self.env.action_space.list_of_actions)>0:
            actions = []
            potential_reward_max = -20
            for action in self.env.action_space.list_of_actions:
                ae = action.evaluate(self.env.current_state_of_the_game)
                potential_reward = (np.floor((current_points + ae["card"][2])/POINTS_TO_WIN)*100 + 10 * ae["card"][2] + 2 *ae["nobles"] + ae["card"][0] + self.weight * sum(ae["gems_flow"]))
                if potential_reward > potential_reward_max:
                    potential_reward_max = potential_reward
                    actions = []
                    actions.append(action)
                elif potential_reward == potential_reward_max:
                    actions.append(action)

            return random.choice(actions)

        else:
            return None
