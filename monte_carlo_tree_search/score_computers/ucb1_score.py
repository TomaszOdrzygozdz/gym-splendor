import math
from monte_carlo_tree_search.constants import INFINITY
from monte_carlo_tree_search.score_computers.abstract_score_computer import ScoreComputer


class UCB1Score(ScoreComputer):

    def __init__(self, exploration_coefficient):
        self.exploration_coefficient = exploration_coefficient

    def compute_score(self, node, parent):

        n_i = node.value_acc.count()
        N_i = parent.value_acc.count()

        assert N_i > 0, 'Parents count equal 0'
        exploitation_term = node.value_acc.get()

        if n_i > 0 and N_i > 0 and exploitation_term is not None:
            exploration_term = self.exploration_coefficient*math.sqrt(math.log(N_i)/n_i)
            return exploitation_term + exploration_term

        else:
            return INFINITY