# import logging, os
# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from keras.layers import Lambda
import keras.backend as K

def card_noble_mask(inputs):
  cards = inputs[0]
  mask = inputs[1]

  dotted = K.batch_dot(cards, mask, axes=[-2, -1])
  mask_sum = K.sum(mask, axis=-1, keepdims=True)
  results = tf.math.divide(dotted, mask_sum)
  return results

def tensor_squeeze(inputs):
  result = tf.squeeze(inputs, axis=1)
  return result

CardNobleMasking = Lambda(card_noble_mask)
TensorSqueeze = Lambda(tensor_squeeze)



