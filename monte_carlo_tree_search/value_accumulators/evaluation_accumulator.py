from monte_carlo_tree_search.value_accumulators.abstract_value_accumulator import ValueAccumulator


class EvaluationAccumulator(ValueAccumulator):

    def __init__(self, state=None, mean_max_coeff=1):

        super().__init__(None)
        self._eval = None
        self._perfect_value = None
        self._count = 0

    def add_count(self):
        if self._perfect_value is None:
            assert self._eval is not None, 'Cannot increase count without previous evaluation'
            self._count += 1

    def count(self):
        return self._count

    def get(self):
        if self._perfect_value is not None:
            return self._perfect_value
        else:
            if self._eval is not None:
                return self._eval
            else:
                return None

    def add_eval(self, value):
        stop_backprop = False

        if self._eval is not None:
            if value >= self._eval:
                self._eval = value
            if value < self._eval:
                stop_backprop = True

        if self._eval is None:
            self._eval = value

        self._count += 1
        return stop_backprop

    def set_constant_value_for_terminal_node(self, perfect_value):
        self._perfect_value = perfect_value
        self._count = 1