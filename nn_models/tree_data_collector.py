import random

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

    def setup_last_vertex(self, vertex : DeterministicTreeNode):
        self.last_vertex = vertex

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

    def generate_all_tree_data_main_track(self, confidence_threshold: float  = 0.1, count_threshold: int = 6, confidence_limit: int=2):
        self.clean_memory()
        print('Collecting tree data.')
        X = []
        Y = []
        vertex = self.last_vertex
        if len(vertex.children) > 0:
            child_idx = random.randint(0,len(vertex.children)-1)
            child_state = StateAsDict(vertex.children[child_idx].return_state()).to_state()
            child_state_inversed = StateAsDict(vertex.children[child_idx].return_state()).to_state()
            child_state_inversed.change_active_player()
            X.append(child_state)
            Y.append(vertex.children[child_idx].value_acc.get())
            X.append(child_state_inversed)
            Y.append(-vertex.children[child_idx].value_acc.get())
        while vertex.parent is not None:
            # take first:
            vertex_state = StateAsDict(vertex.return_state()).to_state()
            vertex_state_inversed = StateAsDict(vertex.return_state()).to_state()
            vertex_state_inversed.change_active_player()
            vertex_value = vertex.value_acc.get()
            X.append(vertex_state)
            Y.append(vertex_value)
            X.append(vertex_state_inversed)
            Y.append(-vertex_value)
            vertex = vertex.parent
        return {'state': X, 'mcts_value' : Y}

    def generate_all_tree_data_as_list(self, confidence_threshold: float = 0.1, count_threshold: int = 6,
                                       confidence_limit: int = 2):
        self.clean_memory()
        # BFS
        kiu = [self.root]
        confidence_count = 0
        print('Collecting tree data.')
        X = []
        Y = []
        while len(kiu) > 0:
            # take first:
            node_to_eval = kiu.pop(0)
            if node_to_eval.value_acc.count() > 0:
                for child in node_to_eval.children:
                    # if child.value_acc.count() >= count_threshold or child.value_acc.perfect_value is not None:
                    if child.value_acc.get_confidence() >= confidence_threshold:
                        confidence_count += 1
                    if (
                            child.value_acc.get_confidence() >= confidence_threshold and confidence_count <= confidence_limit) \
                            or child.value_acc.count() >= count_threshold:
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
