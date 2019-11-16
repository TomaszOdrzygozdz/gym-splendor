from typing import Dict


class StateAsDict:

    def __init__(self, state_as_dict : Dict):
        self.state_as_dict = state_as_dict

    def __getitem__(self, item):
        return self.state_as_dict[item]

    def clone(self):
        return StateAsDict(self.state_as_dict.copy())
