"""Network interface and its implementations."""

import gin

from alpaca.alpacka.networks import core
from alpaca.alpacka.networks import keras


# Configure networks in this module to ensure they're accessible via the
# alpacka.networks.* namespace.
def configure_network(network_class):
    return gin.external_configurable(
        network_class, module='alpacka.networks'
    )


Network = core.Network  # pylint: disable=invalid-name
DummyNetwork = configure_network(core.DummyNetwork)  # pylint: disable=invalid-name
KerasNetwork = configure_network(keras.KerasNetwork)  # pylint: disable=invalid-name
