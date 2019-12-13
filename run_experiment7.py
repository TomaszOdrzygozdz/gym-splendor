from monte_carlo_tree_search.self_play_trainer import SelfPlayTrainer

fufu = SelfPlayTrainer('dqn', iteration_limit=4, choose_best=0.5)
fufu.full_training(10, 0.2, 2)