import tensorflow as tf
import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Layer
from keras.utils import plot_model

from nn_models.utils.named_tuples import *

class GemsInput:
    def __init__(self):
        self.inputs = [Input(shape=(1,), name='gem_{}'.format(color).replace('GemColor.', '')) for color in GemColor]
    def __call__(self, gems_tuple):
        return [np.array([x]) for x in gems_tuple]

class GemsEmbedding:
    def __init__(self, output_dim):
        self.inputs = [Input(shape=(1,), name='gem_{}'.format(color)) for color in GemColor]
        self.color_embeddings = [Embedding(input_dim=6,
                                                              name='embd_gem_{}'.format(color),
                                                              output_dim=output_dim)(self.inputs[color.value])
                                                    for color in GemColor]

        self.layer = Model(inputs = self.inputs, outputs = self.color_embeddings, name = 'gems_embed')

########################################################################################################################

class PriceInput:
    def __init__(self):
        self.inputs = [Input(shape=(1,), name='price_{}'.format(color).replace('GemColor.', '')) for color in GemColor if
                       color != GemColor.GOLD]
    def __call__(self, price_tuple):
        return [np.array([x]) for x in price_tuple]

class PriceEmbedding:
    def __init__(self, output_dim):
        self.inputs = PriceInput().inputs
        self.color_embeddings = [Embedding(input_dim=25,
                                                              name='embd_gem_{}'.format(color),
                                                              output_dim=output_dim)(self.inputs[color.value-1])
                                                    for color in GemColor if color != GemColor.GOLD]

        self.layer = Model(inputs = self.inputs, outputs = self.color_embeddings, name = 'price_embed')
########################################################################################################################

class CardInput:
    def __init__(self):
        self.inputs = [Input(shape=(1,), name='card_profit')] + PriceInput().inputs +  [Input(shape=(1,), name='points')]
    def __call__(self, card_tuple : CardTuple):
        return [card_tuple.profit] + list(card_tuple.price) + [card_tuple.victory_points]

class CardEmbedding:
    def __init__(self):
        self.inputs = CardInput().inputs



dane = PriceTuple(1, 2, 3, 4, 5)

aaa = PriceInput()
real_inputs = aaa.inputs
bbb = PriceEmbedding(2)
next_layer = bbb.layer(real_inputs)

real_model = Model(inputs = real_inputs, outputs = next_layer, name='gems_embedder' )
optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
real_model.compile(optimizer=optim, loss='mean_squared_error')
#wynik = real_model.predict(x = aaa(dane))
plot_model(real_model, to_file='real_model.png', show_shapes=True)
#print(wynik)
#         def __call__(self, gems_tuple):
#
#
#
#
# class MiniModelsHolder:
#     def __init__(self):
#         self.gems_embedding = GemsEmebedding(10)
#
#
# class Summer(keras.layers.Layer):
#     def __init__(self):
#         pass
#
#     def __call__(self, ll):










# class CardEmbedder:
#
#     def _call__(self, list_of_card_tuples, mask):
#         #card = (a b c)
#         a_s
#         b_s
#         c_s
        


