from monte_carlo_tree_search.value_accumulators.abstract_value_accumulator import ValueAccumulator


class ScalarMeanMaxValueAccumulator(ValueAccumulator):

    def __init__(self, value=-1, state=None, mean_max_coeff=1):

        self._sum = 0.0
        self._count = 0
        self._max = value
        self.mean_max_coeff = mean_max_coeff
        self.evaluation = None
        self.perfect_value = None
        self.confidence_counter = 0
        super().__init__(value, state)

    def add(self, value, high_confident_value: bool = False):
        self._max = max(self._max, value)
        self._sum += value
        self._count += 1
        if high_confident_value:
            self.confidence_counter += 1

    def set_constant_value_for_terminal_node(self, perfect_value):
        self.perfect_value = perfect_value
        self._count = 1

    def get_confidence(self):
        if self.perfect_value is not None:
            return 1
        if self.count() > 0:
            return self.confidence_counter / self.count()
        return 0

    def get(self):
        if self.perfect_value is not None:
            return self.perfect_value
        if self._count > 0:
            return (self._sum / self._count)*self.mean_max_coeff \
               + self._max*(1-self.mean_max_coeff)
        else:
            return 0

    def index(self, parent_value=None, action=None):
        return self.get() + self.auxiliary_loss  # auxiliary_loss alters tree traversal in monte_carlo_tree_search

    def final_index(self, parent_value=None, action=None):
        return self.index(parent_value, action)

    def target(self):
        return self.get()

    def count(self):
        return self._count