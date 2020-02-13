# import os
#
from training_data.data_generation.gen_data_lvl0 import produce_data, load_data_for_model, pick_data_for_training
#
# #produce_data(1, 0.25, 10000, 'all_vs_all.pickle', '/net/archive/groups/plggluna/plgtodrzygozdz/lvl1/train_raw')
#
pick_data_for_training(range(0, 20), '/net/archive/groups/plggluna/plgtodrzygozdz/lvl1/train_raw',
                       '/net/archive/groups/plggluna/plgtodrzygozdz/lvl1/train_epochs')
#
# #data = load_data_for_model('/home/tomasz/ML_Research/splendor/gym-splendor/cluster_stuff/temp/proc_205_all_vs_all.pickle')
#
# #list_of_states = len(data[0])
# #print(len(data))
#
# #x = os.listdir('/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/lvl0')
