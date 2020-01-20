import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import keras
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Embedding, Concatenate, Dense
from keras.utils import plot_model


from archive.states_list import state_3
from nn_models.utils.named_tuples import *
from nn_models.utils.vectorizer import Vectorizer

class GemsEncoder:
    def __init__(self, output_dim):
        self.inputs = [Input(batch_shape=(None, None, 1), name='gem_{}'.format(color)) for color in GemColor]
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

class CardEncoder:
    def __init__(self, profit_dim, price_dim, points_dim, dense1_dim, dense2_dim,  max_points=25):
        self.price_encoder = PriceEncoder(output_dim=price_dim)
        self.inputs = [Input(batch_shape=(None, None, 1), name='card_profit'), *self.price_encoder.inputs,
                       Input(batch_shape=(None, None, 1), name='victory_points')]
        profit_embedded = Embedding(input_dim=5, output_dim=profit_dim, name='profit')(self.inputs[0])
        price_encoded = self.price_encoder.layer(self.inputs[1:-1])
        price_concatenated = Concatenate(axis=-1)(price_encoded)
        points_embedded = Embedding(input_dim=max_points, output_dim=points_dim, name='points')(self.inputs[-1])
        full_card = Concatenate(axis=-1)([profit_embedded, price_concatenated, points_embedded])
        full_card  = Dense(units=dense1_dim)(full_card)
        full_card = Dense(units=dense2_dim)(full_card)
        self.layer = Model(inputs = self.inputs, outputs = full_card, name = 'card_encoder')

    def __call__(self, card_input_list):
        return self.layer(card_input_list)

class NobleEncoder:
    def __init__(self, price_dim, dense1_dim, dense2_dim):
        self.price_encoder = PriceEncoder(output_dim=price_dim)
        self.inputs = self.price_encoder.inputs
        price_encoded = self.price_encoder.layer(self.inputs)
        price_concatenated = Concatenate(axis=-1)(price_encoded)
        

class BoardEncoder:
    def __init__(self, gems_dim, profit_dim, price_dim, points_dim, dense1_dim, dense2_dim):
        self.gems_encoder = GemsEncoder(gems_dim)
        self.card_encoder = CardEncoder(profit_dim, price_dim, points_dim, dense1_dim, dense2_dim)
        self.inputs = self.gems_encoder.inputs + self.card_encoder.inputs





card_encoder = CardEncoder(1, 1, 1, 2, 2)
model_inputs = [Input(batch_shape=(None, None, 1)) for _ in range(7)]
model_outputs = card_encoder.layer(model_inputs)
real_model = Model(inputs=model_inputs, outputs=model_outputs, name='real_model')
optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
real_model.compile(optimizer=optim, loss='mean_squared_error')
plot_model(real_model, to_file='tf2.0_real_model.png')
card = state_3.board.cards_on_board.pop()

def card_to_tensor(card_tuple: CardTuple):
    pre_list = [np.array(card_tuple.profit)] + [np.array(x) for x in card_tuple.price] + \
               [np.array(card_tuple.victory_points)]
    list_to_return = []
    for x in pre_list:
        list_to_return.append(x.reshape(1,1,1))
    return list_to_return

xxx = card_to_tensor(Vectorizer().card_to_tuple(card))

wyn = real_model.predict(x=xxx)
print(wyn)

