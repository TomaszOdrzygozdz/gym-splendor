import time
from nn_models.value_function_heura.value_evaluator import ValueFunctionOptimizer
from agents.value_function_agent import ValueFunctionAgent

from mpi4py import  MPI

from agents.greedy_agent_boost import GreedyAgentBoost
from arena.arena_multi_thread import ArenaMultiThread

main_process = MPI.COMM_WORLD.Get_rank() == 0

do = 1


if do == 1:
    moominer = ValueFunctionOptimizer()
    time_s = time.time()
    val = moominer.eval_metrics(100)
    if main_process:
        print(f'Time taken = {time.time() - time_s}')
        print(f'value  = {val}')

if do == 2:
    a1 = GreedyAgentBoost()
    a2 = ValueFunctionAgent()
    arek = ArenaMultiThread()
    res = arek.run_many_games('deterministic', [a1, a2], 100)
    if main_process:
        print(res)

if do == 3:
    from agents.random_agent import RandomAgent
    from agents.value_function_agent import ValueFunctionAgent

    a1 = RandomAgent()
    a2 = ValueFunctionAgent()

    from arena.arena import Arena

    arek = Arena()
    arek.run_one_duel('deterministic', [a1, a2], render_game=True)