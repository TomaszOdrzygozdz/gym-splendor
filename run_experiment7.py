from agents.random_agent import RandomAgent
from arena.multi_arena import MultiArena
from monte_carlo_tree_search.self_play_trainer import SelfPlayTrainer

import cProfile

staty = cProfile.Profile()

fufu = SelfPlayTrainer('dqn', iteration_limit=24, rollout_repetition=2, choose_best=0.2)
staty.run('fufu.full_training(10, 0.1, 2)')
staty.dump_stats('self_play.prof')