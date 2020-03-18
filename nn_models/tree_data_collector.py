from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from monte_carlo_tree_search.trees.deterministic_tree import DeterministicTreeNode
import pandas as pd

from archive.vectorization import vectorize_state

class TreeDataCollector:

    def __init__(self):
        self.clean_memory()

    def setup_root(self, root : DeterministicTreeNode, jump_to_parent=True):
        self.root = root
        if jump_to_parent:
            if self.root.parent is not None:
                self.root = self.root.parent

    def generate_dqn_data(self):
        self.clean_memory()
        self.generate_all_tree_data()

    def generate_all_tree_data(self):
        self.clean_memory()
        all_code = ''
        # BFS
        kiu = [self.root]

        print('Collecting tree data.')

        while len(kiu) > 0:
            # take first:
            node_to_eval = kiu.pop(0)
            if node_to_eval.value_acc.count() > 0:
                for child in node_to_eval.children:
                    if child.value_acc.count() > 0:
                        kiu.append(child)
                        child_state_as_dict = StateAsDict(child.return_state())
                        self.stats_dataframe = self.stats_dataframe.append({'state': child_state_as_dict.to_state(),
                                                                            'mcts_value': child.value_acc.get()},
                                                                           ignore_index=True)
        return self.stats_dataframe

    def generate_all_tree_data_as_list(self, count_threshold: int  = 2):
        self.clean_memory()
        # BFS
        kiu = [self.root]

        print('Collecting tree data.')
        X = []
        Y = []
        while len(kiu) > 0:
            # take first:
            node_to_eval = kiu.pop(0)
            if node_to_eval.value_acc.count() > 0:
                for child in node_to_eval.children:
                    if child.value_acc.count() >= count_threshold or child.value_acc.perfect_value is not None:
                        kiu.append(child)
                        child_state_as_dict = StateAsDict(child.return_state())
                        current_state = child_state_as_dict.to_state()
                        current_state_inversed = child_state_as_dict.to_state()
                        current_state_inversed.change_active_player()
                        current_value = child.value_acc.get()

                        X.append(current_state)
                        Y.append(current_value)
                        X.append(current_state_inversed)
                        Y.append(-current_value)



        return {'state': X, 'mcts_value' : Y}

    def dump_data(self, file_name):
        self.stats_dataframe.to_csv(file_name + '_.pickle', header=True)
        self.clean_memory()

    def return_data(self):
        return self.stats_dataframe

    def clean_memory(self):
        self.stats_dataframe = pd.DataFrame(columns=('state', 'mcts_value'))
