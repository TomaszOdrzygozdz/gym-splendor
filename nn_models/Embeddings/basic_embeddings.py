import tensorflow as tf
import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Concatenate
from keras.utils import plot_model

from archive.states_list import state_3
from nn_models.utils.named_tuples import *
from nn_models.utils.vectorized_old import Vectorizer

class GemsEncoder:
    def __init__(self, output_dim):
        self.inputs = [Input(batch_shape=(None, None, 1), name='gem_{}'.format(color)) for color in GemColor]
        self.color_embeddings = [Embedding(input_dim=6,
                                                              name='embd_gem_{}'.format(color),
                                                              output_dim=output_dim)(self.inputs[color.value])
                                                    for color in GemColor]

        self.layer = Model(inputs = self.inputs, outputs = self.color_embeddings, name = 'gems_embed')

    def __call__(self, list_of_gems):
        return self.layer(list_of_gems)

# ########################################################################################################################

class PriceEncoder:
    def __init__(self, output_dim):
        self.inputs = PriceInput().inputs
        self.color_embeddings = [Embedding(input_dim=25,
                                                              name='embd_gem_{}'.format(color),
                                                              output_dim=output_dim)(self.inputs[color.value-1])
                                                    for color in GemColor if color != GemColor.GOLD]

#         self.layer = Model(inputs = self.inputs, outputs = self.color_embeddings, name = 'price_embed')
# ########################################################################################################################
#
# class CardInput:
#     def __init__(self):
#         self.inputs = [Input(shape=(1,), name='card_profit'), PriceInput().inputs, Input(shape=(1,), name='victory_points')]
#     def __call__(self, card_tuple : CardTuple):
#         return [card_tuple.profit, [x for x in card_tuple.price], card_tuple.victory_points]
#
# class CardEmbedding:
#     def __init__(self, card_profit_dim=2, gem_dim=3, points_dim=3):
#
#         self.inputs = CardInput().inputs
#         self.price_embedding_layer = PriceEmbedding(gem_dim).layer
#         profit_embedded = Embedding(input_dim=5, output_dim=card_profit_dim, name='card_profit')(self.inputs[0])
#         price_embedded = self.price_embedding_layer(self.inputs[1])
#         points_embedded = Embedding(input_dim=25, output_dim=points_dim, name='points')(self.inputs[2])
#         card_embedded = [profit_embedded, profit_embedded, points_embedded]
#
#         self.layer = Model(inputs = self.inputs, outputs=card_embedded, name='card_embedding')
#
#
# card = state_3.board.cards_on_board.pop()
# converter = Vectorizer()
# card_t = converter.card_to_tuple(card)
# print(card_t)
#
# card_input = CardInput()
# model_inputs = card_input.inputs
# card_embedding = CardEmbedding(2, 2, 2)
# card_output = card_embedding.layer(card_input.inputs)
#
# real_model = Model(inputs = card_inputs, outputs = card_output, name='gems_embedder' )
# optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# real_model.compile(optimizer=optim, loss='mean_squared_error')
# #wynik = real_model.predict(x = aaa(dane))
# plot_model(real_model, to_file='real_model.png', show_shapes=True)
# #print(wynik)
# #         def __call__(self, gems_tuple):
# #
# #
# #
# #
# # class MiniModelsHolder:
# #     def __init__(self):
# #         self.gems_embedding = GemsEmebedding(10)
# #
# #
# # class Summer(keras.layers.Layer):
# #     def __init__(self):
# #         pass
# #
# #     def __call__(self, ll):
#
#
#
#
#
#
#
#
#
#
# # class CardEmbedder:
# #
# #     def _call__(self, list_of_card_tuples, mask):
# #         #card = (a b c)
# #         a_s
# #         b_s
# #         c_s
        


### test:

# fuf = GemsEncoder(3)
#
# real_model_inputs = [Input(batch_shape=(None, 1)) for _ in range(6)]
# real_outputs = fuf(real_model_inputs)
#
# new_model = Model(inputs = real_model_inputs, outputs=real_outputs)
# optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# new_model.compile(optimizer=optim, loss='mean_squared_error')
# #
#
# plot_model(new_model, to_file='gems_encoder.png', show_shapes=True)
#
# g0 = [np.array([3])]*6
# g1 =  [np.array([[3],[4]])]*6
# print(g1)
# #g1 = np.reshape(1, 2, 6)
#
#
#
# wyn = new_model.predict(x=g1)
# for x in wyn:
#     print('___')
#     print(x)
#     print('____')
#
#
# print('******************************************************************')
#
# wyn = new_model.predict(x=g0)
# for x in wyn:
#     print('___')
#     print(x)
#     print('____')