from typing import List

from agent import Agent
from arena.one_vs_many_statistics import OneVsManyStatistics


class ManyVsManyStatistics:

    def __init__(self, list_of_agents : List[Agent])->None:

        self.list_of_agents = list_of_agents
        self.data = {}
        for agent in list_of_agents:
            opponents = [opponent for opponent in list_of_agents if opponent != agent]
            self.data[agent] = OneVsManyStatistics(agent, opponents)

    def __repr__(self):
        str_to_return = 'Collective results :\n'
        for agent in self.list_of_agents:
            str_to_return += self.data[agent].__repr__()
            str_to_return += '__________________________ \n'
        return str_to_return
