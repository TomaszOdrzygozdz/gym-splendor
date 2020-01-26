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



