import random

from agent import Agent
from gym_splendor_code.envs.mechanics.action import Action


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
        Agent.agents_created += 1
        self.distribution = distribution
        #we create own gym-splendor enivronemt to have access to its functionality
        #We specify the name of the agent
        self.name = 'RandomAgent - ' + self.distribution + ' ' + str(Agent.agents_created)

    def choose_action(self, observation) -> Action:

        #first we load observation to the private environment
        self.env.load_observation(observation)
        self.env.update_actions()

        if len(self.env.action_space.list_of_actions):
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

        else:
            return None
