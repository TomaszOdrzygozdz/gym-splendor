# tree node
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from monte_carlo_tree_search.trees.abstract_tree import TreeNode
from monte_carlo_tree_search.value_accumulators.scalar_min_max_value_acc import ScalarMeanMaxValueAccumulator

class DeterministicTreeNode(TreeNode):

    def __init__(self, observation: DeterministicObservation, parent: 'MCTSTreeNode', parent_action: Action,
                 reward: int, is_done: bool, winner_id: int)->None:
        super().__init__(parent, parent_action, ScalarMeanMaxValueAccumulator())
        assert observation.name == 'deterministic', 'Wrong observation'
        self.state_recreated = False
        #self.observation = DeterministicObservation(self.state)
        self.observation = observation
        self.reward = reward
        self.is_done = is_done
        self.winner_id = winner_id
        self.terminal = False
        if self.winner_id is not None:
            self.solved = True
        if self.is_done:
            self.perfect_value = reward
            self.value_acc.set_constant_value_for_terminal_node(perfect_value=reward)
            self.value_acc.add(reward)
            self.terminal = True


    def active_player_id(self):
        self.recreate_state()
        return self.state.active_player_id

    def recreate_state(self):
        if not self.state_recreated:
            self.state = self.observation.recreate_state()
            self.state_recreated = True

    def check_if_terminal(self):
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
        self.recreate_state()
        if len(self.actions) == 0:
            self.actions = generate_all_legal_actions(self.state)
        self.check_if_terminal()

    def return_state(self):
        self.recreate_state()
        return self.state

    def expanded(self):
        self.recreate_state()
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


