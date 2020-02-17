# from training_data.data_generation.gen_data_lvl0 import evaluate_states
#
# evaluate_states('/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/lvl3/train_epochs',
#                 '/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/lvl3/train_epochs_eval')

from training_data.data_generation.gen_data_lvl0 import evaluate_states

evaluate_states('/net/archive/groups/plggluna/plgtodrzygozdz/lvl1/train_epochs',
                '/net/archive/groups/plggluna/plgtodrzygozdz/lvl1/valid_epoch/valid_eval.pickle')