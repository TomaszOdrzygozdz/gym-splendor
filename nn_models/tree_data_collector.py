from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from monte_carlo_tree_search.trees.abstract_tree import TreeNode
from monte_carlo_tree_search.trees.deterministic_tree import DeterministicTreeNode
import pandas as pd

from nn_models.vectorization import vectorize_state


class TreeDataCollector:

    def __init__(self):
        self.clean_memory()

    def setup_rooot(self, root : DeterministicTreeNode):
        self.root = root

    def generate_dqn_data(self):
        self.clean_memory()
        self.generate_all_tree_data(raw=False, vectorized=True)
        return self.stats_dataframe_vectorized

    def generate_all_tree_data(self, raw:bool =True, vectorized: bool = True):
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
                        if raw:
                            self.stats_dataframe = self.stats_dataframe.append({'state': child_state_as_dict,
                                                                                'mcts_value': child.value_acc.get()},
                                                                               ignore_index=True)

                        if vectorized:
                            self.stats_dataframe_vectorized = self.stats_dataframe_vectorized .append({'state_vector': vectorize_state(child_state_as_dict),
                                                                            'mcts_value': child.value_acc.get()},
                                                                           ignore_index=True)

    def dump_data(self, file_name):
        self.stats_dataframe.to_csv(file_name + '_raw.csv', header=True)
        self.stats_dataframe_vectorized.to_csv(file_name + '_vectorized.csv', header=True)
        self.clean_memory()


    def clean_memory(self):
        self.stats_dataframe = pd.DataFrame(columns=('state', 'mcts_value'))
        self.stats_dataframe_vectorized = pd.DataFrame(columns=('state_vector', 'mcts_value'))