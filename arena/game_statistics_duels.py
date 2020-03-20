from itertools import product
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from agents.abstract_agent import Agent
from arena.one_agent_statistics import OneAgentStatistics


class GameStatisticsDuels:

    def __init__(self,
                 list_of_agents1: List[Agent] = None,
                 list_of_agents2: List[Agent] = None,
                 number_of_games: int = 0) -> None:

        if list_of_agents2 is None:
            list_of_agents2 = list_of_agents1

        #create lists of agents names:
        self.list_of_agents_names1 = [agent.my_name_with_id() for agent in list_of_agents1]
        self.list_of_agents_names2 = [agent.my_name_with_id() for agent in list_of_agents2]
        pairs = list(product(self.list_of_agents_names1, self.list_of_agents_names2))
        reversed_pairs = list(product(self.list_of_agents_names2, self.list_of_agents_names1))
        all_pairs = pairs
        for pair in reversed_pairs:
            if pair not in pairs:
                all_pairs.append(pair)

        self.data = {pair: OneAgentStatistics() for pair in all_pairs if pair[0] != pair[1]}
        self.n_games_dict = {pair: 0 for pair in all_pairs}
        self.number_of_games = number_of_games

    def register_from_dict(self, results_dict: Dict):
        list_agents_names = list(results_dict.keys())
        #product of agents:
        if len(results_dict.keys()) > 0:
            for i in range(2):
                if len(list_agents_names) < 2:
                    assert len(list_agents_names) >= 2, results_dict
                else:
                    self.data[(list_agents_names[i], list_agents_names[(i+1)%2])] = results_dict[list_agents_names[i]]
                    self.n_games_dict[(list_agents_names[i], list_agents_names[(i + 1) % 2])] += 1
        self.number_of_games += 1

    def return_stats(self):
        if len(self.list_of_agents_names1) < 2:
            pair =  (self.list_of_agents_names1[0], self.list_of_agents_names2[0])
        else:
            pair = (self.list_of_agents_names1[0], self.list_of_agents_names1[1])
        return self.number_of_games, self.data[pair].reward, self.data[pair].wins, self.data[pair].victory_points


    def register(self, other):
        if other is not None:
            for pair in other.data.keys():
                self.data[pair] = self.data[pair] + other.data[pair]
                self.n_games_dict[pair] = self.n_games_dict[pair] + other.n_games_dict[pair]
            self.number_of_games += other.number_of_games


    def to_pandas(self, param='wins', average: bool=True, crop_names:bool = True, n_games = 1000):
        list_of_all_agents_names = self.list_of_agents_names1
        list_of_all_agents_names.extend(agent_name for agent_name in self.list_of_agents_names2
                                        if agent_name not in self.list_of_agents_names1 )


        data_frame = pd.DataFrame()
        for pair in self.data:
            name1 = pair[0]
            name2 = pair[1]
            if crop_names:
                name1 = self.crop_name(name1)
                name2 = self.crop_name(name2)
            data_to_record = {'wins': self.data[pair].wins,
                              'reward': self.data[pair].reward,
                              'victory_points': self.data[pair].victory_points,
                              'games': self.n_games_dict[pair]}

            entry = data_to_record[param]

            if average and self.n_games_dict[pair] > 0 and param != "games":
                entry = round(float(entry/self.n_games_dict[pair]), 5)
            elif average and self.n_games_dict[pair] > 0 and param == "games":
                entry = round(float(entry/n_games/2), 5)
            data_frame.loc[self.crop_name(pair[0]), self.crop_name(pair[1])] = entry

        data_frame.sort_index(inplace=True)
        return data_frame

    def create_heatmap(self, param='wins', average: bool = True, p1 = 15, p2 = 2, n_games = 1000):
        data_frame = self.to_pandas(param, average, n_games = n_games)
        data_frame.sort_index(inplace=True, ascending = False)
        data_frame = data_frame.reindex(sorted(data_frame.columns), axis=1)
        plt.figure(figsize=(p1, p1))
        sns.set(font_scale=p2)
        return sns.heatmap(data_frame, annot=True, fmt='g')

    def __repr__(self):
        str_to_return = '\n {} games taken: \n'.format(self.number_of_games)
        for pair in list(self.data.keys()):
            str_to_return += '[' + pair[0] + '] AGAINST [' + pair[1] + ']: ' + self.data[pair].__repr__() + '\n'
        return str_to_return

    def crop_name(self, name):
        return name[-4:]

