import gin
import numpy as np

@gin.configurable
class EnsembleConfigurator:

    def __init__(self, num_ensembles=1):
        self.num_ensembles = num_ensembles

@gin.configurable
class InferenceEnsembleMembers:
    """ Keeps how many ensemble member should be used per trajectory.

    Value can be passed as integer by `num_members`, or as ratio of ensemble
    size by `ratio` parameter.
    """
    def __init__(self, num_members=None, ratio=1):
        assert num_members is None or ratio is None
        self._num_members = num_members
        self.ratio = ratio

    def num_members(self, ensemble_size=None):
        if self._num_members:
            return self._num_members
        else:
            assert ensemble_size > 0
            print(
                f"num_members {int(np.ceil(self.ratio * ensemble_size))}"
            )
            return int(np.ceil(self.ratio * ensemble_size))
