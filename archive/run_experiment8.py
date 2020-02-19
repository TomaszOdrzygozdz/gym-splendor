import matplotlib.pyplot as plt
import pickle
import math

with open('/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/lvl3/train_epochs_eval/epoch_9.pickle', 'rb') as f:
    _, Y = pickle.load(f)

new_Y = []
for y in Y:
    new_Y.append(y**0.125)
plt.hist(new_Y, bins = 200)
plt.show()