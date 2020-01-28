# import logging, os
# logging.disable(logging.WARNING)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

EPSILON = 0.00000001

import tensorflow as tf

from keras.layers import Lambda
import keras.backend as K

# def card_noble_mask(inputs):
#   cards = inputs[0]
#   mask = inputs[1]
#
#   dotted = K.batch_dot(cards, mask, axes=[-2, -1])
#   mask_sum = K.sum(mask, axis=-1, keepdims=True)
#   results = tf.math.divide(dotted, mask_sum)
#   return results

def card_noble_mask(inputs):
  cards = inputs[0]
  mask = inputs[1]
  assert len(cards.shape) == 3, f'Got shape {cards.shape}'
  assert len(mask.shape) == 2, f'Got shape {mask.shape}'

  cards_count = tf.reduce_sum(mask, axis=1)
  cards_count = tf.expand_dims(cards_count, axis=-1)
  mask = tf.expand_dims(mask, axis=-1)
  cards = cards*mask
  cards_sum = tf.reduce_sum(cards, axis=1)
  cards_averages = tf.divide(cards_sum, (cards_count + EPSILON))

  assert len(cards_averages.shape) == 2, f'Got shape {cards_averages.shape}'

  return cards_averages

# card = (batch, seq, dim)
# mask = (batch, seq)

def tensor_squeeze(inputs):
  result = tf.squeeze(inputs, axis=1)
  return result

CardNobleMasking = Lambda(card_noble_mask)
TensorSqueeze = Lambda(tensor_squeeze)



