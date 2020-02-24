import time
from additional_experiments.gradient_estimation import ValueFunctionOptimizer

from mpi4py import  MPI
main_process = MPI.COMM_WORLD.Get_rank() == 0

do = 1

if do == 1:
    moominer = ValueFunctionOptimizer()
    time_s = time.time()
    val = moominer.eval_metrics(100)
    if main_process:
        print(f'Time taken = {time.time() - time_s}')
        print(f'value  = {val}')

else:
    from agents.random_agent import RandomAgent
    from agents.value_function_agent import ValueFunctionAgent

    a1 = RandomAgent()
    a2 = ValueFunctionAgent()

    from arena.arena import Arena

    arek = Arena()
    arek.run_one_duel('deterministic', [a1, a2], render_game=True)