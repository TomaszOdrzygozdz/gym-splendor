"""Network interface implementation using the Keras framework."""

import functools

import gin
import tensorflow as tf
from tensorflow import keras

from alpacka.networks import core


@gin.configurable
def mlp(input_shape, hidden_sizes=(32,), activation='relu',
        output_activation=None):
    """Simple multilayer perceptron."""
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for h in hidden_sizes:
        x = keras.layers.Dense(h, activation=activation)(x)
    outputs = keras.layers.Dense(
        # 1 output hardcoded for now (value networks).
        # TODO(koz4k): Lift this restriction.
        1,
        activation=output_activation,
        name='predictions',
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs)


@gin.configurable
def convnet_mnist(
    input_shape,
    n_conv_layers=5,
    d_conv=64,
    d_ff=128,
    activation='relu',
    output_activation=None,
):
    """Simple convolutional network."""
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(n_conv_layers):
        x = keras.layers.Conv2D(
            d_conv, kernel_size=(3, 3), padding='same', activation=activation
        )(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(d_ff, activation=activation)(x)
    outputs = keras.layers.Dense(
        # 1 output hardcoded for now (value networks).
        # TODO(koz4k): Lift this restriction.
        1,
        activation=output_activation,
        name='predictions',
    )(x)
    return keras.Model(inputs=inputs, outputs=outputs)


class KerasNetwork(core.Network):
    """Network implementation in Keras.

    Args:
        input_shape (tuple): Input shape.
        model_fn (callable): Function input_shape -> tf.keras.Model.
        optimizer: See tf.keras.Model.compile docstring for possible values.
        loss: See tf.keras.Model.compile docstring for possible values.
        weight_decay (float): Weight decay to apply to parameters.
        metrics: See tf.keras.Model.compile docstring for possible values
            (Default: None).
        train_callbacks: List of keras.callbacks.Callback instances. List of
            callbacks to apply during training (Default: None)
        **compile_kwargs: These arguments are passed to tf.keras.Model.compile.
    """

    def __init__(
        self,
        input_shape,
        model_fn=mlp,
        optimizer='adam',
        loss='mean_squared_error',
        weight_decay=0.0,
        metrics=None,
        train_callbacks=None,
        **compile_kwargs
    ):
        super().__init__(input_shape)
        self._model = model_fn(input_shape)
        self._add_weight_decay(self._model, weight_decay)
        self._model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=metrics or [],
                            **compile_kwargs)

        self.train_callbacks = train_callbacks or []

    @staticmethod
    def _add_weight_decay(model, weight_decay):
        # Add weight decay in form of an auxiliary loss for every layer,
        # assuming that the weights to be regularized are in the "kernel" field
        # of every layer (true for dense and convolutional layers). This is
        # a bit hacky, but still better than having to add those losses manually
        # in every defined model_fn.
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                # Keras expects a parameterless function here. We use
                # functools.partial instead of a lambda to workaround Python's
                # late binding in closures.
                layer.add_loss(functools.partial(
                    keras.regularizers.l2(weight_decay), layer.kernel
                ))

    def train(self, data_stream):
        """Performs one epoch of training on data prepared by the Trainer.

        Args:
            data_stream: (Trainer-dependent) Python generator of batches to run
                the updates on.

        Returns:
            dict: Collected metrics, indexed by name.
        """

        dataset = tf.data.Dataset.from_generator(
            generator=data_stream,
            output_types=(self._model.input.dtype, self._model.output.dtype)
        )

        # WA for bug: https://github.com/tensorflow/tensorflow/issues/32912
        history = self._model.fit_generator(dataset, epochs=1, verbose=0,
                                            callbacks=self.train_callbacks)
        # history contains epoch-indexed sequences. We run only one epoch, so
        # we take the only element.
        return {name: values[0] for (name, values) in history.history.items()}

    def predict(self, inputs):
        """Returns the prediction for a given input.

        Args:
            inputs: (Agent-dependent) Batch of inputs to run prediction on.

        Returns:
            Agent-dependent: Network predictions.
        """

        return self._model.predict_on_batch(inputs).numpy()

    @property
    def params(self):
        """Returns network parameters."""

        return self._model.get_weights()

    @params.setter
    def params(self, new_params):
        """Sets network parameters."""

        self._model.set_weights(new_params)

    def save(self, checkpoint_path):
        """Saves network parameters to a file."""

        self._model.save_weights(checkpoint_path, save_format='h5')

    def restore(self, checkpoint_path):
        """Restores network parameters from a file."""

        self._model.load_weights(checkpoint_path)
