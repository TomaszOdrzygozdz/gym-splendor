import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Embedding
from keras.utils import plot_model

class ComputeSum(keras.layers.Layer):
  def __init__(self):
    super(ComputeSum, self).__init__()

  def call(self, l):
    return tf.math.add(l[0], l[1])


inpA = Input(shape=(None,3), name='A')
inpB = Input(shape=(None,4), name='B')
fA = Dense(3, activation='relu')(inpA)
fB = Dense(3, activation='relu')(inpB)

susu = ComputeSum()([fA, fB])

internal = Model(inputs=[inpA, inpB], outputs = susu, name='internal_model')
optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#neti.compile(optimizer=optim, loss='mean_squared_error')
#plot_model(neti, to_file='model_archie_larp.png', show_shapes=True, show_layer_names='True')

inpC = Input(shape=(2,), name='C')
inpD = Input(shape=(2,), name='D')
inpE = Input(shape=(2,), name='E')
inpF = Input(shape=(2,), name='F')


x1 = Dense(3, activation='relu')(inpC)
x2 = Dense(4, activation='relu')(inpD)
x3 = Dense(3, activation='relu')(inpE)
x4 = Dense(4, activation='relu')(inpF)

fuf1 = internal([x1, x2])
fuf2 = internal([x3, x4])

fuf2b = Dense(3, activation='relu', name='dupa')(fuf2)

kupidynek = ComputeSum()([fuf1, fuf2b])

master_model = Model(inputs = [inpC, inpD, inpE, inpF], outputs = kupidynek)
master_model.compile(optimizer=optim, loss='mean_squared_error')
plot_model(master_model, to_file='kupidynek.png')



# v1 = np.array([1,1,1])
# v2 = np.array([0,0,0,0])
# v1 = v1.reshape(1,3)
# v2 = v2.reshape(1,4)
#
# u1 = np.array([0,0,0])
# u2 = np.array([1,2,3,4])
# u1 = u1.reshape(1,3)
# u2 = u2.reshape(1,4)
#
# # o1 = neti.predict([v1, v2])
# # print(o1)
# # o2 = neti.predict([u1, u2])
# # print(o2)
# #
# # o3 = neti.predict([v1, u2])
# # print(o3)
# #
