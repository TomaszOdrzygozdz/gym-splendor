import csv
import random
import numpy as np
from copy import copy
import os

import elopy
from tqdm import tqdm

from arena.game_statistics_duels import GameStatisticsDuels

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ELO_RATINGS_FILE = os.path.join(DATA_DIR, 'elo_ratings.csv')

class LeaderBoard:

    def __init__(self, list_of_agents):
        self.elo_implementation = elopy.Implementation()
        for agent in list_of_agents:
            self.elo_implementation.addPlayer(agent.my_name_with_id())

    def load_from_file(self, file=ELO_RATINGS_FILE):
        elo_rating_file = open(file)
        reader = csv.reader(elo_rating_file)
        for row in reader:
            player = self.elo_implementation.getPlayer(row[0])
            player.rating = float(row[1])

    def register_from_games_statistics(self, games_statistics: GameStatisticsDuels):
        list_of_wins = []
        for entry in games_statistics.data:
            list_of_wins += [entry]*(games_statistics.data[entry].wins)
        random.shuffle(list_of_wins)
        for entry in tqdm(list_of_wins):
            self.elo_implementation.recordMatch(entry[0], entry[1], winner=entry[0], draw=False)

    def get_rating_list(self):
        return self.elo_implementation.getRatingList()

    def save_to_file(self, file=ELO_RATINGS_FILE):
        np.savetxt(file, self.elo_implementation.getRatingList(), delimiter=',', fmt='%s')

    def __repr__(self):
        str_to_return = '\n Leader board: \n'
        for entry in self.elo_implementation.getRatingList():
            str_to_return += ' {}: {} \n'.format(entry[0], round(entry[1],1))
        return str_to_return

