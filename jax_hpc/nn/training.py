import jax
import jaxlib
import optax
import haiku as hk
from typing import Iterator, NamedTuple, Callable, Tuple, Dict
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp
from ..nn.models import *


class Batch(NamedTuple):
    """Image/label batch. Taken from
    https://github.com/google-deepmind/dm-haiku/blob/main/examples/mnist.py
    """

    image: np.ndarray  # [B, H, W, C]
    label: np.ndarray  # [B]


class TrainingState(NamedTuple):
    """Model/Optimizer training state. Taken from
    https://github.com/google-deepmind/dm-haiku/blob/main/examples/mnist.py
    """

    params: hk.Params
    opt_state: optax.OptState


def image_cat_cross_entropy(
    params: hk.Params, batch: Batch, model: Callable, num_classes: int = 10
) -> jax.Array:
    """Mean batch cross-entropy classification loss. Taken from
    https://github.com/google-deepmind/dm-haiku/blob/main/examples/mnist.py

    Parameters
    ----------
    params:
        Instance of `hk.Params` for a model
    batch:
        `Batch` instance containing inputs and labels
    num_classes:
        Number of classes for classification

    Returns
    -------
    loss:
        Batchwise negative log likelihood
    """

    batch_size, *_ = batch.image.shape
    logits = model.apply(params, batch.image)
    labels = jax.nn.one_hot(batch.label, num_classes)
    log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))

    return -log_likelihood / batch_size


class ClassifierTrainer(object):
    """Basic wrapper for training

    Parameters
    ----------
    model:
        HaikuModel instance storing hyperparemters and possessing a `generate_haiku_module`
        method for creating Haiku transforms
    optimizer:
        Instanced OPTAX optimizer
    loss_fn:
        Callable that computes training/validation losses. IMPORTANT: given the default behavior
        of `jax.grad`, the first argument to `loss_fn` must be an instance of `hk.Params`. See
        https://jax.readthedocs.io/en/latest/_autosummary/jax.grad.html for more information.
    verbose:
        If `True`, model training/validation progress will be printed to STDOUT during
        training.
    """

    def __init__(
        self,
        model: HaikuModel,
        optimizer: optax._src.base.GradientTransformation,
        loss_fn: Callable,
        verbose: bool = True,
    ):
        # Model forward kernels must be cached using
        # haiku transforms of generated haiku modules
        self.model = hk.without_apply_rng(
            hk.transform(lambda x: model.generate_haiku_model()(x.astype(jnp.float32)))
        )
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_classes = model.hyperparams["out_dim"]
        self.initialized = False
        self.verbose = verbose

    def validate(
        self, params: hk.Params, batch: Batch, num_classes: int = 10
    ) -> Tuple[jax.Array, jax.Array]:
        """Predicts batchwise validation loss/accuracy using supplied parameters

        Parameters
        ----------
        params:
            Instance of `hk.Params` for a model
        batch:
            `Batch` instance containing inputs and labels
        num_classes:
            Number of classes for classification

        Returns
        -------
        loss:
            Batchwise loss
        accuracy:
            Batchwise accuracy
        """
        loss = jnp.mean(
            self.loss_fn(params, batch, self.model, num_classes=num_classes)
        )
        logits = self.model.apply(params, batch.image)
        predictions = jnp.argmax(logits, axis=-1)
        return jnp.mean(loss), jnp.mean(predictions == batch.label)

    def update(
        self, state: TrainingState, batch: Batch, num_classes: int = 10
    ) -> Tuple[jax.Array, TrainingState]:
        """Gradient update over batch of training examples, returns loss and new
        parameter/optimizer state

        Parameters
        ----------
        state:
            `TrainingState` instance containing model parameters and an optimizer
            state
        batch:
            `Batch` instance containing inputs and labels
        num_classes:
            Number of classes for classification

        Returns
        -------
        loss:
            Batchwise loss
        updated_state:
            `TrainingState` with optimizer-updated model parameters and a new
            optimizer state
        """

        # get gradients from model predictions and targets via the loss function
        grads = jax.grad(self.loss_fn)(state.params, batch, self.model, num_classes)
        # get gradient updates and new optimizer state
        updates, opt_state = self.optimizer.update(grads, state.opt_state)

        # get gradient-updated params
        params = optax.apply_updates(state.params, updates)
        return jnp.mean(
            self.loss_fn(state.params, batch, self.model, num_classes)
        ), TrainingState(params, opt_state)

    def set_initial_state(self, dummy_input: np.ndarray):
        """Initilizes model parameters and optimizer state

        Parameters
        ----------
        dummny_input:
            `np.ndarray` dummy input for use in haiku transform `init` method
        """

        initial_params = self.model.init(jax.random.PRNGKey(seed=0), dummy_input)
        initial_opt_state = self.optimizer.init(initial_params)
        self.current_training_state = TrainingState(initial_params, initial_opt_state)
        self.initialized = True

    @staticmethod
    def dataset_to_iterator(
        ds: tf.data.Dataset, batch_size=int, shuffle: bool = True
    ) -> Iterator:
        """Method for transforming a tensorflow dataset into an Iterator over
        `Batch` instances

        Parameters
        ----------
        ds:
            input `tf.data.Dataset` instance
        batch_size:
            integer batch size for resulting iterator
        shuffle:
            if `True`, the data will be shuffled before being returned

        Returns
        -------
        iterator:
            `Iterator` over `Batch`es of size batch_size
        """

        ds.cache()
        if shuffle:
            ds.shuffle(batch_size)
        ds = ds.unbatch()
        ds = ds.batch(batch_size)
        ds = ds.map(lambda x: Batch(**x))
        return iter(tfds.as_numpy(ds))

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        num_epochs: int = 300,
    ):
        """Training/validation loop

        Parameters
        ----------
        train_dataset:
            Tensorflow dataset containing data for training
        val_dataset:
            Tensorflow dataset containing data for validation.
            Validation is performed at the end of every epoch.
        num_epochs:
            Number of passes through the entire training dataset
        """

        if not self.initialized:
            raise RuntimeError(
                "Model and optimizer not initialized. Call `Trainer.set_initial_state` first."
            )

        for epoch in range(num_epochs):
            train_examples = self.dataset_to_iterator(
                train_dataset, batch_size=512, shuffle=True
            )
            val_examples = self.dataset_to_iterator(
                train_dataset, batch_size=512, shuffle=False
            )
            train_losses = []
            validation_losses = []
            validation_accuracy = []
            for i, train_batch in enumerate(train_examples):
                loss, self.current_training_state = self.update(
                    self.current_training_state,
                    train_batch,
                    num_classes=self.num_classes,
                )
                train_losses.append(loss)
            for i, validation_batch in enumerate(val_examples):
                loss, accuracy = self.validate(
                    self.current_training_state.params,
                    validation_batch,
                    num_classes=self.num_classes,
                )
                validation_losses.append(loss)
                validation_accuracy.append(accuracy)
            if self.verbose:
                jax.debug.print(
                    f"Epoch {epoch}: [train {jnp.mean(np.array(train_losses)):.4f}] --> [val {jnp.mean(np.array(validation_losses)):.4f}] --> [val acc {jnp.mean(np.array(validation_accuracy)):.4f}]"
                )