import jax
import jaxlib
import jax.numpy as jnp
import haiku as hk
from typing import Callable, List, Optional


class HaikuModel(object):
    """Abstract custom Haiku Module"""

    def __init__(self, *args, **kwargs):
        pass

    def generate_haiku_model(self, *args):
        raise NotImplementedError()


class FullyConnectedClassifier(HaikuModel):
    """Fully connected, feed-forward image classifier.
    Implements a factory for the downstream production of
    pure forwards for JAX.

    Parameters
    ----------
    out_dim:
        `int` specifying the number of classes
    activation:
        Valid tf.keras.activations `str` or `Callable`
        instance for the hidden layer activations
    hidden_layers:
        `List[int]` of hidden layer dimensions for linear transforms
    """

    def __init__(
        self,
        out_dim: int,
        activation: Callable,
        hidden_layers: Optional[List[int]] = None,
    ):
        super(FullyConnectedClassifier, self).__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        self.hyperparams = {
            "out_dim": out_dim,
            "activation": activation,
            "hidden_layers": hidden_layers,
        }

    def generate_haiku_model(self) -> hk.Module:
        """Factory method for generating Haiku Modules
        suitable for instatiation within haiku.transform scopes
        for JAX kernel caching.
        """
        layers = []
        layers.append(hk.Flatten())
        layers.append(hk.Linear(self.hyperparams["hidden_layers"][0]))
        layers.append(self.hyperparams["activation"])
        if len(self.hyperparams["hidden_layers"]) > 1:
            for i in range(1, len(self.hyperparams["hidden_layers"])):
                layers.append(hk.Linear(self.hyperparams["hidden_layers"][i]))
                layers.append(self.hyperparams["activation"])
        layers.append(hk.Linear(self.hyperparams["out_dim"]))

        net = hk.Sequential(layers)
        return net


class ModelWrapper(object):
    """Wrapper for config instantiations"""

    def __init__(self, model: HaikuModel):
        self.model = model
