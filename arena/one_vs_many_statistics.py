from typing import List

from agent import Agent


class OneVsManyStatistics:

    def __init__(self, who_plays : Agent, list_of_opponents: List[Agent])->None:
        self.name = who_plays.name
        self.who_plays = who_plays
        self.list_of_opponents = list_of_opponents
        self.reward_dict = {opponent.my_name_with_id(): 0 for opponent in list_of_opponents}
        self.win_dict = {opponent.my_name_with_id(): 0 for opponent in list_of_opponents}

    def register(self, opponent, reward, win):
        assert opponent in self.list_of_opponents, 'Opponent not in self.list_of_opponents.'
        dict_key = opponent.my_name_with_id
        self.reward_dict[dict_key] = reward
        self.win_dict[dict_key] = win

    def __add__(self, other):
        assert self.name == other.name, 'Cannot add results of different agents.'
        assert self.list_of_opponents == other.list_of_opponent, 'List of opponents must be qual'
        new_statistics = OneVsManyStatistics(self.who_plays, self.list_of_opponents)
        new_statistics.reward_dict = {opponent.my_name_with_id() : self.reward_dict[opponent.my_name_with_id()] +
                                      other.reward_dict[opponent.my_name_with_id()] for opponent
                                      in self.list_of_opponents}
        new_statistics.win_dict = {opponent.my_name_with_id(): self.win_dict[opponent.my_name_with_id()] +
                                                                  other.win_dict[opponent.my_name_with_id()] for
                                      opponent
                                      in self.list_of_opponents}
        return new_statistics

    def __repr__(self):
        str_to_return = self.name + ' against: \n'
        for opponent in self.list_of_opponents:
            str_to_return += '[' + opponent.name + ' | reward: {}, win: {} ] \n'.\
                format(self.reward_dict[opponent.my_name_with_id()], self.win_dict[opponent.my_name_with_id()])
        return str_to_return
