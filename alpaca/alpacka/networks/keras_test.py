"""Tests for alpacka.networks.keras."""

import os
import tempfile

import numpy as np
import pytest
from tensorflow import keras

from alpacka.networks import keras as keras_networks


@pytest.fixture
def keras_mlp():
    return keras_networks.KerasNetwork(input_shape=(13,))


@pytest.fixture
def dataset():
    ((x_train, y_train), _) = keras.datasets.boston_housing.load_data()
    return (x_train, y_train)


@pytest.mark.parametrize('model_fn,input_shape,output_shape', [
    (keras_networks.mlp, (15,), (1,)),
    (keras_networks.convnet_mnist, (3, 3, 6), (1,)),
])
def test_model_valid(model_fn, input_shape, output_shape):
    network = keras_networks.KerasNetwork(
        model_fn=model_fn, input_shape=input_shape
    )
    batch_size = 7
    inp = np.zeros((batch_size,) + input_shape)
    out = network.predict(inp)
    assert out.shape == (batch_size,) + output_shape


def test_keras_mlp_train_epoch_on_boston_housing(keras_mlp, dataset):
    # Set up
    (x_train, y_train) = dataset
    x_train = x_train[:16]
    y_train = y_train[:16]

    def data_stream():
        for _ in range(3):
            yield (x_train, y_train)

    # Run
    metrics = keras_mlp.train(data_stream)

    # Test
    assert 'loss' in metrics


def test_keras_mlp_predict_batch_on_boston_housing(keras_mlp, dataset):
    # Set up
    (data, _) = dataset
    data_batch = data[:16]

    # Run
    pred_batch = keras_mlp.predict(data_batch)

    # Test
    assert pred_batch.shape == (16, 1)


def test_keras_mlp_modify_weights(keras_mlp):
    # Set up
    new_params = keras_mlp.params
    for p in new_params:
        p *= 2

    # Run
    keras_mlp.params = new_params

    # Test
    for new, mlp in zip(new_params, keras_mlp.params):
        assert np.all(new == mlp)


def test_keras_mlp_save_weights(keras_mlp):
    # Set up, Run and Test
    with tempfile.NamedTemporaryFile() as temp_file:
        assert os.path.getsize(temp_file.name) == 0
        keras_mlp.save(temp_file.name)
        assert os.path.getsize(temp_file.name) > 0


def test_keras_mlp_restore_weights(keras_mlp):
    with tempfile.NamedTemporaryFile() as temp_file:
        # Set up
        orig_params = keras_mlp.params
        keras_mlp.save(temp_file.name)

        new_params = keras_mlp.params
        for p in new_params:
            p *= 2
        keras_mlp.params = new_params

        # Run
        keras_mlp.restore(temp_file.name)

        # Test
        for orig, mlp in zip(orig_params, keras_mlp.params):
            assert np.all(orig == mlp)
