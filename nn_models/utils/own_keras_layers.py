class Concatenate(keras.layers.Layer):
  def __init__(self):
    super(ComputeSum, self).__init__()

  def call(self, l):
    return tf.math.add(l[0], l[1])
