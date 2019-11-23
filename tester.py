#from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict


sum = State()


di = sum.to_dict()

print(di.state_as_dict)

muk = State(load_from_state_as_dict=di, prepare_state=True)

print(muk.to_dict().state_as_dict)
