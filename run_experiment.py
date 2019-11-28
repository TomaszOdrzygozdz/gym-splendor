#from experiments.vanilla_mcts_experiments.vanilla_mcts_experiment_v1 import run


from arena.multi_process.multi_arena import DeterministicMultiProcessArena
arek = DeterministicMultiProcessArena()

arek.run_many_duels(['a', 'b'], 10, 2)


# from experiments.baseline_comparison.baseline_comparison_v1 import run_baseline_comparison_v1
#
# run_baseline_comparison_v1()
