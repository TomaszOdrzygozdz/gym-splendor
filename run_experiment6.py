from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from nn_models.dense_model import DenseModel



fuf = DenseModel()
fuf.create_network()
fuf.load_weights(weights_file='E:\ML_research\gym_splendor\\nn_models\weights\minmax_480_games.h5')
stan = State()
actio = generate_all_legal_actions(stan)[1]
q_v = fuf.get_q_value(StateAsDict(stan), actio)
print(q_v)

