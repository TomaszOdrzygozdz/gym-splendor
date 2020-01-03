
import os
import io
import pandas as pd

from nn_models.value_dense_model import ValueDenseModel



class ValueSupervisedTrainer:

    def __init__(self):

        self.model = ValueDenseModel()
        self.layers_list = [600, 600, 600, 600]
        self.model.create_network(layers_list=self.layers_list)


    def get_model_summary(self):
        stream = io.StringIO()
        self.model.network.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        return summary_string

    def merge_data(self, folder_path = 'states_train_data', output = None):
        list_of_files = os.listdir(folder_path)
        df_list = [pd.read_pickle(os.path.join(folder_path, file)) for file in list_of_files]
        collected_df = pd.concat(df_list)
        if output is not None:
            collected_df.to_pickle(output)

        return collected_df



fufix = ValueSupervisedTrainer()
#
# fufix.merge_data(output='merged.pi')

bubik = ValueDenseModel()
bubik.create_network(layers_list=[500, 500, 500])
bubik.train_model(data_file_name='merged.pi', output_weights_file_name='wuwik.h5')