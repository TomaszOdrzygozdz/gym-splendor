from training_data.data_generation.gen_data_lvl0 import produce_data, pick_data_for_training, evaluate_states


def produce_large_data():
    #produce_data(0,0.1, 150000, 'large', '/net/archive/groups/plggluna/plgtodrzygozdz/large_states_set/raw_states')
    # pick_data_for_training(range(15), '/net/archive/groups/plggluna/plgtodrzygozdz/large_states_set/raw_states',
    #                        '/net/archive/groups/plggluna/plgtodrzygozdz/large_states_set/picked_states', 5)
    evaluate_states('/net/archive/groups/plggluna/plgtodrzygozdz/large_states_set/picked_states',
                    '/net/archive/groups/plggluna/plgtodrzygozdz/large_states_set/evaluated_data')