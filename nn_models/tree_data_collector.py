from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from monte_carlo_tree_search.trees.abstract_tree import TreeNode
from monte_carlo_tree_search.trees.deterministic_tree import DeterministicTreeNode
import pandas as pd

from nn_models.vectorization import vectorize_state


class TreeDataCollector:

    def __init__(self, root : DeterministicTreeNode):
        self.root = root
        self.stats_dataframe = pd.DataFrame(columns=('state', 'value'))
        self.stats_dataframe_vectorized = pd.DataFrame(columns=('state_vector', 'value'))
        self.generate_tree_data()

    def generate_tree_data(self):
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
                        child_state_as_dict = StateAsDict(child.state)
                        self.stats_dataframe = self.stats_dataframe.append({'state': child_state_as_dict,
                                                                            'value': child.value_acc.get()},
                                                                           ignore_index=True)

                        self.stats_dataframe_vectorized = self.stats_dataframe_vectorized .append({'state_vector': vectorize_state(child_state_as_dict),
                                                                        'evaluation': child.value_acc.get()},
                                                                       ignore_index=True)

    def dump_data(self, file_name, clean=True):
        self.stats_dataframe.to_csv(file_name + '_raw.csv', header=True)
        self.stats_dataframe_vectorized.to_csv(file_name + '_vectorized.csv', header=True)
        if clean:
            self.stats_dataframe = pd.DataFrame(columns=('state', 'value'))
            self.stats_dataframe_vectorized = pd.DataFrame(columns=('state_vector', 'value'))


