# tree node
import gym

from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from monte_carlo_tree_search.trees.abstract_tree import TreeNode
from monte_carlo_tree_search.value_accumulators.scalar_min_max_value_acc import ScalarMeanMaxValueAccumulator

#TODO Finish

class DeterministicTreeNodeFastX(TreeNode):
    env = gym.make('splendor-v0')

    def __init__(self, parent: 'MCTSTreeNode', parent_action: Action)->None:


        super().__init__(parent, parent_action, ScalarMeanMaxValueAccumulator(), )
        self.state_calculated = False
        self.observation = None
        self.reward = 0
        self.winner_id = None
        self.is_done = False
        self.terminal = False
        #self.observation = DeterministicObservation(self.state)

    def load_observation(self, observation):
        print('Creating root')
        self.observation = observation
        self.state = self.observation.recreate_state()
        self.state_calculated = True

    def calculate_state(self):
        if not self.state_calculated:
            print('Calculating')
            if not self.is_root():
                self.env.load_observation(self.parent.observation)
                observation, reward, is_done, info = self.env.step('deterministic', self.parent_action)
                self.observation = observation
                self.perfect_value = None
                self.reward = reward
                self.winner_id = info['winner_id']
                self.is_done = is_done
                self.state = self.observation.recreate_state()
                self.state_calculated = True
                self.terminal = False
                if self.winner_id is not None:
                    self.solved = True
                if self.is_done:
                    self.perfect_value = reward
                    self.terminal = True

    def active_player_id(self):
        self.calculate_state()
        return self.state.active_player_id

    def check_if_terminal(self):
        self.calculate_state()
        #Node can be terminal in two ways
        if not self.terminal:
            if self.is_done:
                self.teminal = True
            else:
                self.terminal = terminal = True if len(self.actions) == 0 else False
                if self.winner_id is None and self.terminal:
                    self.winner_id = self.state.previous_player_id()
        else:
            pass

    def generate_actions(self):
        self.calculate_state()
        if len(self.actions) == 0:
            self.actions = generate_all_legal_actions(self.state)
        self.check_if_terminal()

    def state(self):
        self.calculate_state()
        return self.state

    def expanded(self):
        self.calculate_state()
        if self.terminal:
            return True
        else:
            return True if self.actions else False

    def parent(self):
        return self.parent

    def is_root(self):
        if self.parent is None:
            return True
        else:
            return False


