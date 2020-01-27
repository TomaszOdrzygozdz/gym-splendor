from neptune_settings import USE_NEPTUNE
import tensorflow as tf
from keras import backend as K
from keras.utils import plot_model

if USE_NEPTUNE:
    from neptune_settings import NEPTUNE_API_TOKEN, NEPTUNE_PROJECT_NAME_NN_TRAINING
    import  neptune


class AbstractModel:

    def __init__(self):
        self.network = None
        self.params = {}



    def start_neptune_experiment(self, project_name=NEPTUNE_PROJECT_NAME_NN_TRAINING, experiment_name='Experiment',
                                 description = ' ', neptune_monitor=None):
        neptune.init(project_qualified_name=project_name, api_token=NEPTUNE_API_TOKEN)
        if neptune_monitor is not None:
            self.neptune_monitor = neptune_monitor
        neptune.create_experiment(name=experiment_name, description=description, params=self.params)
        neptune.log_image('Architecture', 'model_architecture.png')

    def model_summary(self):
        assert self.network is not None, 'You must create network before vizualization.'
        plot_model(self.network, to_file='model_architecture.png', show_shapes=True)
        summary_str = []
        self.network.summary(print_fn=lambda x: summary_str.append(x + '\n'))
        short_model_summary = "\n".join(summary_str)
        return short_model_summary

