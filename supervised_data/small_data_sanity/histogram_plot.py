import pickle
import matplotlib.pyplot as plt
from nn_models.value_function_heura.value_function import ValueFunction

with open('/home/tomasz/ML_Research/splendor/gym-splendor/supervised_data/test0/valid_epochs/epoch_0.pickle', 'rb') as f:
    X, _ = pickle.load(f)

vf = ValueFunction()
Y = [vf.evaluate(st) for st in X]
print(len(Y))
print(min(Y))
print(max(Y))
plt.hist(Y, bins=60)
plt.show()

