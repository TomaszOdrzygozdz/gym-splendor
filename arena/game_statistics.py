from typing import Dict

from agent import Agent
from arena.one_agent_statistics import OneAgentStatistics


class GameStatistics:

    def __init__(self,
                 statistics_dict: Dict[Agent, OneAgentStatistics]=None, number_of_games=0) -> None:
        self.dict = statistics_dict
        self.number_of_games = number_of_games

    def create_from_list_of_agents(self, list_of_agents):
        self.dict = {agent.my_name_with_id() : OneAgentStatistics() for agent in list_of_agents}

    def __add__(self,
                other):
        return GameStatistics(
            {agent_name: self.dict[agent_name] + other.dict[agent_name] for agent_name in self.dict.keys()},
        self.number_of_games + other.number_of_games)

    def __truediv__(self, other: float):
        return GameStatistics({agent_name: self.dict[agent_name] / other for agent_name in self.dict.keys()})

    def __repr__(self):
        text_to_return = '\n'
        for agent_name in self.dict.keys():
            text_to_return += '[' + agent_name
            text_to_return += self.dict[agent_name].__repr__()
            text_to_return += '] \n'
        return text_to_return + 'Games: {} \n'.format(self.number_of_games)

