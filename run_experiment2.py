import os

from training_data.data_generation.gen_data_lvl0 import produce_data, load_data_for_model, pick_data_for_training

#produce_data(10, 0.25, 5, 'all_vs_all.pickle', '/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/test_lvl0')

pick_data_for_training(range(0,2), '/net/archive/groups/plggluna/plgtodrzygozdz/lvl3/lvl3',
                       '/net/archive/groups/plggluna/plgtodrzygozdz/lvl3_training')

#data = load_data_for_model('/home/tomasz/ML_Research/splendor/gym-splendor/cluster_stuff/temp/proc_205_all_vs_all.pickle')

#list_of_states = len(data[0])
#print(len(data))

#x = os.listdir('/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/lvl0')
