from typing import List

import gym
from tqdm import tqdm

from monte_carlo_tree_search.rolluot_policy import RolloutPolicy
from monte_carlo_tree_search.tree import TreeNode


class MCTS:
    def __init__(self,
                 iteration_limit: int,
                 rollout_policy: RolloutPolicy,
                 rollout_repetition: int,
                 environment_id: str)->None:
        """:param:
        rollout_policy - object of type RolloutPolicy determing rollout
        iteration_limit - maximum mcts passes for simulation
        rollout_repetition - repetitions of rollout to make in chosen leaf
        environment_id - environment used by the model"""
        if iteration_limit < 1:
            raise ValueError("Iteration limit must be greater than one")
        self.iteration_limit = iteration_limit
        self.rollout_policy = rollout_policy
        self.n_rollout_repetition = rollout_repetition
        self.env = gym.make(environment_id)

    def create_root(self, state_or_observation):
        raise NotImplementedError

    def _expand_leaf(self, leaf: TreeNode):
        raise NotImplementedError

    def _rollout(self, leaf):
        raise NotImplementedError

    def _tree_traversal(self):
        raise NotImplementedError

    def _select_child(self, node: TreeNode):
        raise NotImplementedError

    def _select_best_child(self, node: TreeNode):
        raise NotImplementedError

    def _backpropagate(self, search_path: List[TreeNode], winner_id, value):
        raise NotImplementedError

    def run_mcts_pass(self, rollout_repetition: int = None):
        print('run pass')
        if not self.root.terminal:
            print('not terminal')
            leaf, search_path = self._tree_traversal()
            print('search path chosen')
            rollout_repetition = self.n_rollout_repetition if rollout_repetition is None else rollout_repetition
            print('rollout repetition done')
            for _ in range(rollout_repetition):
                print('before rollout')
                winner_id, value = self._rollout(leaf)
                print('after rollout')
                self._backpropagate(search_path, winner_id, value)
                print('backpropagated')
            print('leaf exanding')
            self._expand_leaf(leaf)
            print('leaf expanded')

    def run_simulation(self, number_of_passes:int=None):
        assert self.root is not None, 'Root is None. Cannot run MCTS pass.'
        number_of_passes = self.iteration_limit if number_of_passes is None else number_of_passes
        for i in tqdm(range(number_of_passes)):
            self.run_mcts_pass()