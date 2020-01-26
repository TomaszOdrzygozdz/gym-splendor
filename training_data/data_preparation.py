import numpy as np
import pandas as pd
import tqdm
import pickle

progress_bar = tqdm.tqdm()
tqdm.tqdm_pandas(progress_bar)
#load data
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from nn_models.utils.vectorizer import Vectorizer

raw_data_small = pd.read_pickle('/home/tomasz/ML_Research/splendor/gym-splendor/training_data/small_merged.pi')
#
vectorizer = Vectorizer()

def obs_to_state(obs : DeterministicObservation):
    return obs.recreate_state()

series_of_states = raw_data_small['observation'].progress_map(obs_to_state)

X_list = series_of_states.tolist()
n = len(X_list)
m = n - n%5
X_list = X_list[0:m]

X = vectorizer.many_states_to_input(X_list)
Y = np.array(raw_data_small['value'].tolist()[0:m]).reshape(m, 1)



with open('small_data.pickle', 'wb') as f:
    pickle.dump({'X' : X, 'Y' : Y}, f)

print(Y.shape)
print(Y[0].shape)
