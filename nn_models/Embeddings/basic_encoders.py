import logging, os
from _ast import Lambda
from copy import deepcopy

from gym_splendor_code.envs.mechanics.game_settings import MAX_CARDS_ON_BORD, MAX_RESERVED_CARDS, \
    NOBLES_ON_BOARD_INITIAL

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import keras
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Embedding, Concatenate, Dense, TimeDistributed
from keras.utils import plot_model


from archive.states_list import state_3
from nn_models.utils.named_tuples import *
from nn_models.utils.vectorizer import Vectorizer

class GemsEncoder:
    def __init__(self, output_dim):
        self.inputs = [Input(batch_shape=(None, 1), name='gems_{}'.format(color).replace('GemColor.', '')) for color in GemColor]
        self.color_embeddings = [Embedding(input_dim=6,
                                                              name='embd_gem_{}'.format(color).replace('GemColor.', ''),
                                                              output_dim=output_dim)(self.inputs[color.value])
                                                    for color in GemColor]
        self.layer = Model(inputs = self.inputs, outputs = self.color_embeddings, name = 'gems_encoder')

    def __call__(self, list_of_gems):
        return self.layer(list_of_gems)

class PriceEncoder:
    def __init__(self, output_dim):
        self.inputs = [Input(batch_shape=(None, None, 1), name='gem_{}'.format(color).replace('GemColor.', '')) for color in GemColor
                       if color != GemColor.GOLD]
        self.price_embeddings = [Embedding(input_dim=25,
                                                              name='embd_gem_{}'.format(color),
                                                              output_dim=output_dim)(self.inputs[color.value-1])
                                                    for color in GemColor if color != GemColor.GOLD]
        self.layer = Model(inputs=self.inputs, outputs=self.price_embeddings, name='price_encoder')
    def __call__(self, list_of_gems):
        return self.layer(list_of_gems)

class ManyCardEncoder:
    def __init__(self, seq_dim, profit_dim, price_dim, points_dim, dense1_dim, dense2_dim,  max_points=25):
        self.price_encoder = PriceEncoder(output_dim=price_dim)
        self.inputs = [Input(batch_shape=(None, seq_dim, 1), name='{}'.format(x)) for x in CardTuple._fields]
        profit_embedded = Embedding(input_dim=5, output_dim=profit_dim, name='profit_embedd')(self.inputs[0])
        price_encoded = self.price_encoder.layer(self.inputs[1:-1])
        price_concatenated = Concatenate(axis=-1)(price_encoded)
        points_embedded = Embedding(input_dim=max_points, output_dim=points_dim, name='points_embedd')(self.inputs[6])
        full_cards = Concatenate(axis=-1)([profit_embedded, price_concatenated, points_embedded])
        full_cards  = Dense(units=dense1_dim)(full_cards)
        full_cards = Dense(units=dense2_dim)(full_cards)
        self.layer = Model(inputs = self.inputs, outputs = full_cards, name = 'card_encoder')
    def __call__(self, card_input_list):
        return self.layer(card_input_list)

class ManyNobleEncoder:
    def __init__(self,price_dim, dense1_dim, dense2_dim):
        self.price_encoder = PriceEncoder(output_dim=price_dim)
        self.inputs = [Input(batch_shape=(None, NOBLES_ON_BOARD_INITIAL, 1), name=x) for x in PriceTuple._fields]
        price_encoded = self.price_encoder.layer(self.inputs)
        price_concatenated = Concatenate(axis=-1)(price_encoded)
        full_noble = Dense(dense1_dim)(price_concatenated)
        full_noble = Dense(dense2_dim)(full_noble)
        self.layer = Model(inputs = self.inputs, outputs = full_noble, name='noble_encoder')
    def __call__(self, noble_input_list):
        return self.layer(noble_input_list)

class BoardEncoder:
    def __init__(self, gems_dim, profit_dim, price_dim, points_dim, dense1_dim, dense2_dim):
        self.gems_encoder = GemsEncoder(gems_dim)
        self.card_encoder = ManyCardEncoder(MAX_CARDS_ON_BORD, profit_dim, price_dim, points_dim, dense1_dim, dense2_dim)
        self.noble_encoder = ManyNobleEncoder(price_dim, dense1_dim, dense2_dim)
        self.inputs = self.gems_encoder.inputs + [Input(batch_shape=(None, MAX_CARDS_ON_BORD, 1), name='card_{}'.format(x)) for x in CardTuple._fields] \
                      + [Input(batch_shape=(None, NOBLES_ON_BOARD_INITIAL, 1), name = 'noble_{}'.format(x)) for x in NobleTuple._fields] #+ \
                      #[Input(batch_shape=(None, 12), name='cards_mask'), Input(batch_shape=(None, 3), name='nobles_mask')]
        gems_input = self.inputs[0:6]
        cards_input = self.inputs[6:13]
        nobles_input = self.inputs[13:18]
        gems_list = self.gems_encoder(gems_input)
        gems_concatenated = Concatenate(axis=-1)(gems_list)
        cards_encoded = self.card_encoder(cards_input)
        cards_reduced = ReduceSequence(cards_encoded)
        nobles_encoded = self.noble_encoder(nobles_input)
        nobles_reduced = ReduceSequence(nobles_encoded)
        full_board = Concatenate(axis=-1)([gems_concatenated, nobles_reduced, cards_reduced])
        # cards_mask = self.inputs[18]
        # nobles_mask = self.inputs[19]
        self.layer = Model(inputs = self.inputs, outputs = full_board, name='board_encoder')
#


bubu = BoardEncoder(1, 2, 2, 2, 10, 11)
plot_model(bubu.layer, to_file='bubu.png', show_shapes=True)
#wyn = bub.layer.predict(x=yuyu[:-2])
#print(wyn)
#
# card_encoder = CardEncoder(3, 1, 1, 1, 32, 2)
# # model_inputs = Input(batch_shape=(None, 1))
# # model_outputs = card_encoder(model_inputs)
# # real_model = Model(inputs=model_inputs, outputs=model_outputs, name='real_model')
# # optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# # real_model.compile(optimizer=optim, loss='mean_squared_error')
# plot_model(card_encoder.layer, to_file='card_encoder_new.png')


# class CardEncoder:
#     def __init__(self, profit_dim, price_dim, points_dim, dense1_dim, dense2_dim,  max_points=25):
#         self.price_encoder = PriceEncoder(output_dim=price_dim)
#         self.inputs = [Input(batch_shape=(None, 1), name='{}'.format(x)) for x in CardTuple._fields]
#         profit_embedded = Embedding(input_dim=5, output_dim=profit_dim, name='profit_embedd')(self.inputs[0])
#         price_encoded = self.price_encoder.layer(self.inputs[1:-1])
#         price_concatenated = Concatenate(axis=-1)(price_encoded)
#         points_embedded = Embedding(input_dim=max_points, output_dim=points_dim, name='points_embedd')(self.inputs[6])
#         full_card = Concatenate(axis=-1)([profit_embedded, price_concatenated, points_embedded])
#         full_card  = Dense(units=dense1_dim)(full_card)
#         full_card = Dense(units=dense2_dim)(full_card)
#         self.layer = Model(inputs = self.inputs, outputs = full_card, name = 'card_encoder')

