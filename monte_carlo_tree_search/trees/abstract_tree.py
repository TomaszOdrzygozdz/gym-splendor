from gym_splendor_code.envs.mechanics.action import Action
from monte_carlo_tree_search.value_accumulators.abstract_value_accumulator import ValueAccumulator


class TreeNode:
    node_id = 0

    def __init__(self,
                 parent: 'MCTSTreeNode',
                 parent_action: Action,
                 value_acc: ValueAccumulator)->None:
        self.node_id = TreeNode.node_id
        TreeNode.node_id += 1
        self.parent = parent
        self.parent_action = parent_action
        self.actions = []
        self.action_to_children_dict = {}
        self.children = []
        self.value_acc = value_acc
        self.perfect_value = None
        if parent is None:
            self.generation = 0
        else:
            self.generation = parent.generation + 1

    def parent(self):
        return self.parent

    def get_id(self):
        return self.node_id

    def check_if_terminal(self):
        raise NotImplementedError

    def is_root(self):
        return True if self.parent is None else False