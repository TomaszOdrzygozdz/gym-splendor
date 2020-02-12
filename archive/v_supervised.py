
import os
import io
import pandas as pd

from archive.dense_models.value_dense_model_v0 import ValueDenseModel



class ValueSupervisedTrainer:

    def __init__(self):
        pass



    def get_model_summary(self):
        stream = io.StringIO()
        self.model.network.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        return summary_string

    def merge_data(self, folder_path = 'raw_value_data', subset=None,  output = None):

        list_of_files = os.listdir(folder_path)
        chosen_files = [os.path.join(folder_path, file) for file in list_of_files if 'judged' in file]
        if subset != None:
            chosen_files = chosen_files[0:subset]
        df_list = [pd.read_pickle(df_file)for df_file in chosen_files]
        print(chosen_files)
        collected_df = pd.concat(df_list)
        if output is not None:
             collected_df.to_pickle(output)

        return collected_df




# fufix = ValueSupervisedTrainer()
# #
# fufix.merge_data(subset=120, output='half_merged.pi')

bubik = ValueDenseModel()
bubik.create_network(layers_list=[400, 400, 400])
bubik.train_model(data_file_name='merged.pi', output_weights_file_name='wuw.h5', experiment_name='Large data train_raw', epochs=50)