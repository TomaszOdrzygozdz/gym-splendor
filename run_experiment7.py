from agents.random_agent import RandomAgent
from arena.multi_arena import MultiArena
from monte_carlo_tree_search.self_play_trainer import SelfPlayTrainer

fufu = SelfPlayTrainer('dqn', iteration_limit=2, rollout_repetition=2, choose_best=0.2)

fufu.full_training(3, 0.1, 2)