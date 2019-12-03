import keras
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Dense, Dropout

import numpy as np
from keras import backend as K


class DQNModel:

    def __init__(self):
        self.session = tf.Session()
        self.network = None

    def set_corrent_session(self):
        K.set_session(self.session)

    def create_network(self, input_size, layers_size_list):
        '''
        This method creates network with a specific architecture
        :return:
        '''
        self.set_corrent_session()
        entries = Input(shape=(598,))
        hidden_vals = Dense(400, activation='relu')(entries)
        hidden_vals = Dense(400, activation='relu')(hidden_vals)
        hidden_vals = Dense(400, activation='relu')(hidden_vals)
        predictions = Dense(1, activation='tanh')(hidden_vals)

        optim = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.network = Model(inputs=entries, outputs=predictions)
        self.network.compile(optimizer=optim, loss='mean_squared_error')

        with open('architecture.txt', 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.network.summary(print_fn=lambda x: fh.write(x + '\n'))
        self.network.summary()

        self.session.run(tf.global_variables_initializer())

    def load_model(self, file_name):
        self.set_corrent_session()
        self.network = load_model(file_name)
        # self.session.run(tf.global_variables_initializer())

    def get_q_value(self, state, move):
        self.set_corrent_session()
        vector_of_state = short_state_to_vector(state)
        vector_of_move = good_move_to_vector(move)
        input_vec = np.array(vector_of_state + vector_of_move)
        return self.network.predict(x=input_vec.reshape(1, 598))[0]

    def find_best_move(self, state):
        self.set_corrent_session()
        list_of_moves = give_all_legal_moves(state, flat=True, check_win_possibility=False)
        if list_of_moves:
            short_vector_of_state = short_state_to_vector(state)
            list_to_eval = vectorize_list_of_moves(short_vector_of_state, list_of_moves)
            q_values_predicted = self.network.predict(x=list_to_eval)
            index_of_best_move = np.argmax(q_values_predicted)
            move_chosen = list_of_moves[index_of_best_move]
            return move_chosen, q_values_predicted[index_of_best_move][0]
        else:
            return None, REWARD_FOR_NO_POSSIBLE_MOVES

    def find_many_best_moves(self, state, N=5):
        self.set_corrent_session()
        list_of_moves = give_all_legal_moves(state, flat=True, check_win_possibility=False)
        N_moves = min(N, len(list_of_moves))
        if list_of_moves:
            short_vector_of_state = short_state_to_vector(state)
            list_to_eval = vectorize_list_of_moves(short_vector_of_state, list_of_moves)
            q_values_predicted = self.network.predict(x=list_to_eval)
            indexes_of_best_move = q_values_predicted.argsort()[-N_moves:]
            moves_chosen = [list_of_moves[int(i)] for i in indexes_of_best_move]
            return moves_chosen, q_values_predicted[indexes_of_best_move]
        else:
            return [None], [REWARD_FOR_NO_POSSIBLE_MOVES]

    def train(self, data_x, data_y, n_epochs=1):
        self.set_corrent_session()
        self.network.fit(np.array(data_x), np.array(data_y), epochs=n_epochs)

    def save(self, file_name):
        self.set_corrent_session()
        self.network.save(file_name)

    # def train_on_winning_moves(self):
    #     data = pd.read_csv('pre_train_data/win_moves_balanced.csv')
    #     data['vector'] = data['vector'].apply(lambda x: pd.json.loads(x))
    #     data_x = np.array(data['vector'].tolist())
    #     data_y = np.array(data['rewards'].tolist())
    #     print(data_x.shape)
    #     print(data_y.shape)
    #     history = self.network.fit(x=data_x, y=data_y, validation_split=0.1, epochs=1, shuffle=True)
    #     self.network.save('pre_train.h5')
    #
    # def collect_data_winning_moves(self, N=100):
    #
    #     cumulated_vectors = []
    #     cumulated_rewards = []
    #     for _ in tqdm(range(N)):
    #         self.prepare_game()
    #         game_is_finished = False
    #         current_state_of_game = vector_to_state(self.vector_of_basic_state)
    #         vectors = []
    #         state_numbers = []
    #         rewards = []
    #         player_points = []
    #         card_id = []
    #         win_id = []
    #         state_number = 0
    #         while not game_is_finished:
    #
    #
    #             move = get_random_action_smart(current_state_of_game)
    #             all_moves, winning_moves = give_all_legal_moves(current_state_of_game, flat=True, check_win_possibility=True)
    #
    #             if len(winning_moves) == 0 and move:
    #                 vectors.append(short_state_to_vector(current_state_of_game) + good_move_to_vector(move))
    #                 rewards.append(0)
    #
    #             if len(winning_moves) > 0:
    #                 for move_to_check in all_moves:
    #                     vectors.append(short_state_to_vector(current_state_of_game) + good_move_to_vector(move_to_check))
    #                     state_numbers.append(state_number)
    #                     win_id.append(winning_moves)
    #                     player_points.append(current_state_of_game.give_active_player().number_of_my_points())
    #                     if move_to_check.move_type == MoveType.RESERVE_CARD or move_to_check.move_type == MoveType.MODIFY_COINS:
    #                         rewards.append(REWARD_FOR_MISSING_WIN)
    #                         card_id.append(-1)
    #                     else:
    #                         if move_to_check.move_type == MoveType.BUY_CARD:
    #                             win = current_state_of_game.win_if_buy_card(move_to_check.id, where_is_card='BOARD')
    #                         if move_to_check.move_type == MoveType.BUY_RESERVED_CARD:
    #                             win = current_state_of_game.win_if_buy_card(move_to_check.id, where_is_card='HAND')
    #                         card_id.append(move_to_check.id)
    #                         if win:
    #                             rewards.append(REWARD_FOR_WINNING_MOVE)
    #                         else:
    #                             rewards.append(REWARD_FOR_MISSING_WIN)
    #
    #
    #             if is_there_winner(current_state_of_game):
    #                 game_is_finished = True
    #
    #             if move is None:
    #                 game_is_finished = True
    #
    #             if state_number > MAX_NUMBER_OF_ROUNDS:
    #                 game_is_finished = True
    #
    #             if current_state_of_game.active_player == PlayerId.FIRST_PLAYER:
    #                 state_number += 1
    #
    #             if move:
    #                 move.execute(current_state_of_game)
    #                 state_number += 1
    #
    #         cumulated_rewards += rewards
    #         cumulated_vectors += vectors
    #     data = pd.DataFrame({'vector' : cumulated_vectors, 'rewards' : cumulated_rewards })
    #     data.to_csv('pre_train_data/win_moves_balanced.csv')
