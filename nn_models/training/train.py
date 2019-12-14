import matplotlib.pyplot as plt

from nn_models.value_dense_model import ValueDenseModel

model = ValueDenseModel()
model.create_network()

history = model.train_model(data_file_name='E:\ML_research\gym_splendor\\nn_models\data\\vectorized\combined.csv',
                  output_weights_file_name='E:\ML_research\gym_splendor\\nn_models\weights\experiment7.h5',
                  epochs = 1)


print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# mumu = DenseNNAgent(weights_file='E:\ML_research\gym_splendor\agents\weights\minmax_480_games.h5')
# opp = RandomAgent(distribution='first_buy')
#
# arek = MultiArena()
# results = arek.run_many_duels('deterministic', [mumu, opp], n_games=20, )


