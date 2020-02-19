# from training_data.data_generation.gen_data_lvl0 import evaluate_states
#
# evaluate_states('/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/lvl3/train_epochs',
#                 '/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/lvl3/train_epochs_eval')

from training_data.data_generation.gen_data_lvl0 import evaluate_states

evaluate_states('/net/archive/groups/plggluna/plgtodrzygozdz/small_data_sanity/train_epochs_eval',
                '/net/archive/groups/plggluna/plgtodrzygozdz/small_data_sanity/valid_eval/valid_eval.pickle')