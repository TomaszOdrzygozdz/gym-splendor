import tensorflow as tf
from keras.layers import Layer, Lambda
#from tensorflow_core.python.ops.gen_array_ops import split


class CardInputSplit(Layer):
  def __init__(self):
    super().__init__()

  def __call__(self, card_to_split):
    profit, red, green, blue, white, black, points = tf.split(axis=-1, value=card_to_split, num_or_size_splits=7,
                                                          name='card_input_split')
    return [profit, red, green, blue, white, black, points]

# class MaskingCards(Layer):
#   def __init__(self):
#     super().__init__()
#     card_encoder
#
#   def call(self, l):
#     return tf.math.add(l[0], l[1])
