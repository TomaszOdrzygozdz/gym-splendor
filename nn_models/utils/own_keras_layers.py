# import logging, os
# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
from keras import Input, Model


from keras.layers import Layer, Lambda, Dense
#from tensorflow_core.python.ops.gen_array_ops import split
from keras.optimizers import Adam
from keras.utils import plot_model
import keras.backend as K

def card_noble_mask(inputs):
  cards = inputs[0]
  mask = inputs[1]
  dotted = K.batch_dot(cards, mask, axes=[-2, -1])
  mask_sum = K.sum(mask, axis=-1)
  results = tf.math.divide(dotted, mask_sum)
  return results

def tensor_squeeze(inputs):
  result = tf.squeeze(inputs, axis=1)
  return result

CardNobleMasking = Lambda(card_noble_mask)
TensorSqueeze = Lambda(tensor_squeeze)
#
# x_in = np.array([[[1, 2, 3], [-1, -2, 6], [-1, -2, 9]]])
# y_in = np.array([1, 1, 1]).reshape(1,3)
# print(y_in.shape)
#
# imp = Input(batch_shape=(None, 1, 3))
# dd = TensorSqueeze(imp)
# ee = Dense(3)(dd)
# #
# fufer = Model(inputs = imp, outputs = ee)
# fufer.compile(Adam(), 'mean_squared_error')
#
# xx = np.array([[1, 2, 3]]).reshape(1, 1, 3)
# print(xx.shape)
# wyn = fufer.predict(xx)
# print(wyn.shape)
#
# print('x = {}'.format(x_in))
# print('scalars =  {}'.format(y_in))
# wyn = fufer.predict(x=[x_in, y_in])
# print('************')
# print(wyn)

# x1 = Input(batch_shape=(None, 3, 4))
# x2 = ReduceSequence(x1)
# fuf = Model(inputs = x1, outputs = x2)
# fuf.compile(Adam(), loss='mean_squared_error')
#
# plot_model(fuf, 'fuf3.png', show_shapes=True)
#
# xx = np.zeros(shape=(1, 3, 4))
# print(xx.shape)
# yy = fuf.predict(x = xx)
# print(yy.shape)

class CardInputSplit(Layer):
  def __init__(self):
    super().__init__()

  def __call__(self, card_to_split):
    profit, red, green, blue, white, black, points = tf.split(axis=-1, value=card_to_split, num_or_size_splits=7,
                                                          name='card_input_split')
    return [profit, red, green, blue, white, black, points]


