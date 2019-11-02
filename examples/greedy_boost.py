from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent, GreedyAgentBoost
from arena import Arena
import time
import random
import numpy as np

fight_pit = Arena()

goku = RandomAgent(distribution='first_buy')
#gohan = RandomAgent(distribution='uniform_on_types')
#gohan = RandomAgent(distribution='uniform')
gohan = GreedyAgent(weight = 0.08)

g1 = GreedyAgentBoost("Greedy1", [100,2,2,1,0.1])
g2 = GreedyAgentBoost("Greedy2", [0,0,0,0,0])
g3 = GreedyAgentBoost("Greedy3", [10,2,2,1,0.2])
g4 = GreedyAgentBoost("Greedy4", [100,0,0,1,0.1])
g5 = GreedyAgentBoost("Greedy5", [0.99954913, 0.01997425, 0.02001405, 0.01004779, 0.00101971])
g6 = GreedyAgentBoost("Greedy6", [0.99953495, 0.02010871, 0.02010487, 0.01095619, 0.00113329])


gv1 = RandomAgent(distribution='first_buy')
gv2 = GreedyAgent(weight = 0.1)
gv3 = GreedyAgentBoost("RandomAgent", [0,0,0,0,0])



g_list = {g1, g2, g3, g4, g5, g6}
gv_list = [gv1, gv2, gv3]
g_list_remove = set()
lr = 0.000005
n_games = 50
max_points = 0
best_weight = [100,2,2,1,0.1]
max_weight = [100,2,2,1,0.1]
best_ratio = 0
most_games = 0
random_player = 0
version = 1

while len(g_list) >  1:
    print('\nExisting players: {}'.format(len(g_list)))
    for gf in g_list:
        #print(gf.name)
        ratios_min = 0.5
        ratios = []
        games = []
        points_rank = []
        points = 0
        for gs in g_list:
            if gf.name is not gs.name:
                #print(gf.name + " vs. " + gs.name)
                result = []
                for i in range(n_games):
                    fight_pit = Arena()
                    turnout = fight_pit.run_one_game([gf, gs], starting_agent_id = 0)
                    if turnout is not None:
                        result.append(turnout)
                #print('Finished games: {}'.format(len(result)))
                res = len(result)
                games.append(res)
                points += len(result) - sum(result)
                if res > 0.5 * n_games:
                    ratio = 1 - np.mean(result)
                    ratios.append(ratio)
                    gf.update_weight([a - b for a, b in zip(gs.weight, gf.weight)], lr*res/n_games, 1 - ratio)
                    gs.update_weight([a - b for a, b in zip(gf.weight, gs.weight)], lr*res/n_games, ratio)
                    #if np.linalg.norm([a - b for a, b in zip(gf.weight, gs.weight)]) < 10e-1:
                    #    print("similarity")
                    #    if ratio > 0.5:
                    #        g_list_remove.add(gs)
                    #    else:
                    #        g_list_remove.add(gf)
        if sum(games) < (len(g_list) - 1) * 0.4 * n_games or points == 0 or ratios_min >= np.mean(ratios):
            #print("small_ratio")
            g_list_remove.add(gf)

        print(gf.name + ":")
        print('Winning ratio: {}'.format(ratios))
        print('Games played: {}'.format(games))
        print('Scored: {}'.format(points))
        print('Weights: {}\n'.format(gf.weight))
        if points /len(g_list) > max_points:
            max_weight = gf.weight
    print('\nRemoving {} players'.format(len(g_list_remove)))
    for i in g_list_remove:
        g_list.remove(i)
        g_list_remove = set()

    print('\nValidation:')

    g_best = GreedyAgentBoost("GreedyBest", max_weight)
    ratios = []
    games = []
    for gvp in gv_list:
        result = []
        for i in range(n_games * 2):
            fight_pit = Arena()
            turnout = fight_pit.run_one_game([g_best, gvp],  starting_agent_id = 0)
            if turnout is not None:
                result.append(turnout)
        print(g_best.name + " vs. " + gvp.name)
        print('Finished games: {}'.format(len(result)))
        print('Winning ratio: {}'.format(1- np.mean(result)))
        ratio = 1 - np.mean(result)
        ratios.append(ratio)
        res = len(result)
        games.append(res)
    print('\nMVP weigts: {}'.format(gv_list[2].weight))
    if np.mean(ratios[0:2]) > best_ratio and ratios[2] >=0.5 and np.mean(games[0:2]) + n_games * 0.1 > most_games:
        best_ratio = np.mean(ratios[0:2])
        most_games = np.mean(games[0:2])
        print('New best weights: {}\n'.format(g_best.weight))
        best_weight = g_best.weight
        g_best.name = "MVP"
        gv_list[2] = g_best
        random_player += 1
        version = 1
    elif len(g_list) < 6:
        print("\nAdding new player")
        g_random = GreedyAgentBoost("GreedyRandom_gen" + str(random_player) + "_" + str(version), best_weight)
        g_random.update_weight( [random.randrange(0,10) for i in range(5)], 0.001, 1)
        g_list.add(g_random)
        version +=1
