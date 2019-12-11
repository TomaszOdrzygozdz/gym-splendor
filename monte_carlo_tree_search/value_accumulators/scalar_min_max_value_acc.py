from monte_carlo_tree_search.value_accumulators.abstract_value_accumulator import ValueAccumulator


class ScalarMeanMaxValueAccumulator(ValueAccumulator):

    def __init__(self, value=0, state=None, mean_max_coeff=0.1):

        self._sum = 0.0
        self._count = 0
        self._max = value
        self.mean_max_coeff = mean_max_coeff
        self.auxiliary_loss = 0.0
        super().__init__(value, state)

    def add(self, value):
        self._max = max(self._max, value)
        self._sum += value
        self._count += 1

    def add_auxiliary(self, value):
        self.auxiliary_loss += value

    def get(self):
        if self._count > 0:
            return (self._sum / self._count)*self.mean_max_coeff \
               + self._max*(1-self.mean_max_coeff)
        else:
            return None

    def index(self, parent_value=None, action=None):
        return self.get() + self.auxiliary_loss  # auxiliary_loss alters tree traversal in monte_carlo_tree_search

    def final_index(self, parent_value=None, action=None):
        return self.index(parent_value, action)

    def target(self):
        return self.get()

    def count(self):
        return self._count