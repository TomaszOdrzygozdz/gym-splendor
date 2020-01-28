import numpy as np
import pandas as pd
import tqdm
import pickle

progress_bar = tqdm.tqdm()
tqdm.tqdm_pandas(progress_bar)
#load data
from gym_splendor_code.envs.mechanics.abstract_observation import DeterministicObservation
from nn_models.utils.vectorizer import Vectorizer

raw_data_small = pd.read_pickle('/home/tomasz/ML_Research/splendor/gym-splendor/training_data/half_merged.pi')
#
vectorizer = Vectorizer()

def obs_to_state(obs : DeterministicObservation):
    return obs.recreate_state()

series_of_states = raw_data_small['observation'].progress_map(obs_to_state)

X_list = series_of_states.tolist()
n = len(X_list)
m = n - n%5
X_list = X_list[0:m]
Y = np.array(raw_data_small['value'].tolist()[0:m]).reshape(m, 1)

indices = [i for i in range(len(Y)) if abs(Y[i])==1]
print(indices)
X_list = [X_list[i] for i in indices]
Y = np.array([Y[i] for i in indices])

print(Y)

X = vectorizer.many_states_to_input(X_list)

with open('half_data_sure.pickle', 'wb') as f:
    pickle.dump({'X' : X, 'Y' : Y}, f)

print(Y.shape)
print(Y[0].shape)
