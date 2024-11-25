import contextlib
from datetime import datetime
import jax
import jaxlib
import optax
import haiku as hk
from typing import Iterator, NamedTuple, Callable, Tuple, Dict
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp
from functools import partial
from ..nn.models import *
import tqdm


class FitWrapper(NamedTuple):
    """Wrapper for config instantiation"""

    num_epochs: int
    # Number of passes through the entire training dataset
    val_freq: int
    # `int` specifying the number of epochs between successive
    # validations


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
    params: hk.Params, batch: Batch, model, num_classes: int = 10
) -> jax.Array:
    """Mean batch cross-entropy classification loss. Taken from
    https://github.com/google-deepmind/dm-haiku/blob/main/examples/mnist.py

    Parameters
    ----------
    params:
        Instance of `hk.Params` for a model
    batch:
        `Batch` instance containing inputs and labels
    model:
        Haiku-transformed model
    lgits:
        `
    num_classes:
        Number of classes for classification

    Returns
    -------
    loss:
        Batchwise negative log likelihood
    """

    logits = model.apply(params, batch.image)
    batch_size, *_ = batch.image.shape
    labels = jax.nn.one_hot(batch.label, num_classes)
    log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))

    return -log_likelihood / batch_size


class ClassifierTrainer(object):
    """Basic wrapper for training a classifier model

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
    logger:
        If specified, a `tf.summary.SummaryWriter` will be used to log a tensorboard output
    proilfer:
        If specified, a `Dict` of profiling options will be used during the training.
    basic_run_stats:
        If `True`, basic performance statistics of the model training will be reported at the end
        of the run.
    """

    def __init__(
        self,
        model: HaikuModel,
        optimizer: optax._src.base.GradientTransformation,
        loss_fn: Callable,
        verbose: bool = True,
        logger: Optional[tf.summary.SummaryWriter] = None,
        profiler: Optional[Dict] = None,
        basic_run_stats: bool = True,
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
        self.logger = logger
        self.profiler = profiler
        self.basic_run_stats = basic_run_stats

    def validate(self, params: hk.Params, batch: Batch) -> Tuple[jax.Array, jax.Array]:
        """Predicts batchwise validation loss/accuracy using supplied parameters

        Parameters
        ----------
        params:
            Instance of `hk.Params` for a model
        batch:
            `Batch` instance containing inputs and labels

        Returns
        -------
        loss:
            Batchwise loss
        accuracy:
            Batchwise accuracy
        """
        loss = jnp.mean(
            self.loss_fn(params, batch, self.model, num_classes=self.num_classes)
        )
        predictions = jnp.argmax(self.model.apply(params, batch.image), axis=-1)
        return jnp.mean(loss), jnp.mean(predictions == batch.label)

    @partial(jax.jit, static_argnums=(0,))
    def update(
        self, state: TrainingState, batch: Batch
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

        Returns
        -------
        loss:
            Batchwise loss
        updated_state:
            `TrainingState` with optimizer-updated model parameters and a new
            optimizer state
        """

        # get gradients from model predictions and targets via the loss function
        grads = jax.grad(self.loss_fn)(
            state.params, batch, self.model, num_classes=self.num_classes
        )
        # get gradient updates and new optimizer state
        updates, opt_state = self.optimizer.update(grads, state.opt_state)

        # get gradient-updated params
        params = optax.apply_updates(state.params, updates)
        return jnp.mean(
            self.loss_fn(state.params, batch, self.model, num_classes=self.num_classes)
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
        ds: tf.data.Dataset, batch_size=int, shuffle: bool = True, cache: bool = False
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
        cache:
            if `True`, the data will be cached (before shuffling)

        Returns
        -------
        iterator:
            `Iterator` over `Batch`es of size batch_size
        """

        if cache:
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
        val_dataset: Optional[tf.data.Dataset],
        num_epochs: int = 300,
        val_freq: int = 10,
        batch_size: int = 512,
        cache: bool = False,
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
        val_freq:
            `int` specifying the number of epochs between successive
             validations
        train_batch_size:
            `int` specifying training/validation batch size
        cache:
            if `True`, shuffled train/val datasets will be cached every epoch
        """

        if not self.initialized:
            raise RuntimeError(
                "Model and optimizer not initialized. Call `Trainer.set_initial_state` first."
            )
        if self.verbose:
            print(f"Training starting: {datetime.now()}")

        with self.logger.as_default() if self.logger is not None else contextlib.nullcontext():
            if self.basic_run_stats:
                train_epoch_times = []
                start_time = None
                end_time = None

                start_time = datetime.now()

            for epoch in range(num_epochs):
                if self.basic_run_stats:
                    train_epoch_start = datetime.now()
                if self.profiler is not None and epoch == self.profiler["start"]:
                    jax.profiler.start_trace(self.profiler["logdir"])

                train_examples = self.dataset_to_iterator(
                    train_dataset, batch_size=512, shuffle=True, cache=cache
                )
                if val_dataset:
                    val_examples = self.dataset_to_iterator(
                        val_dataset, batch_size=512, shuffle=False, cache=cache
                    )

                train_losses = []

                for i, train_batch in tqdm.tqdm(
                    enumerate(train_examples), desc="Training..."
                ):
                    loss, self.current_training_state = self.update(
                        self.current_training_state,
                        train_batch,
                    )
                    train_losses.append(loss)
                if self.basic_run_stats:
                    train_epoch_end = datetime.now()

                if epoch % val_freq == 0 and val_dataset is not None:
                    validation_losses = []
                    validation_accuracy = []
                    for i, validation_batch in tqdm.tqdm(
                        enumerate(val_examples), desc="Validating..."
                    ):
                        loss, accuracy = self.validate(
                            self.current_training_state.params,
                            validation_batch,
                        )
                        validation_losses.append(loss)
                        validation_accuracy.append(accuracy)
                if self.verbose and val_dataset is not None:
                    jax.debug.print(
                        f"Epoch {epoch}: [train {jnp.mean(np.array(train_losses)):.4f}] --> [val {jnp.mean(np.array(validation_losses)):.4f}] --> [val acc {jnp.mean(np.array(validation_accuracy)):.4f}]"
                    )
                if self.verbose and val_dataset is None:
                    jax.debug.print(
                        f"Epoch {epoch}: [train {jnp.mean(np.array(train_losses)):.4f}]"
                    )
                if self.logger is not None:
                    tf.summary.scalar(
                        "training_loss", np.average(np.array(train_losses)), epoch
                    )
                    if val_dataset is not None:
                        tf.summary.scalar(
                            "validation_loss",
                            np.average(np.array(validation_losses)),
                            epoch,
                        )
                        tf.summary.scalar(
                            "validation_accuracy",
                            np.average(np.array(validation_accuracy)),
                            epoch,
                        )
                    self.logger.flush()

                if self.profiler is not None and epoch == self.profiler["stop"]:
                    jax.profiler.stop_trace()

                if self.basic_run_stats:
                    duration = (
                        train_epoch_end.timestamp() - train_epoch_start.timestamp()
                    )
                    train_epoch_times.append(duration)

        if self.basic_run_stats:
            end_time = datetime.now()
            total_duration = end_time.timestamp() - start_time.timestamp()
            avg_train = np.average(train_epoch_times)
            std_train = np.std(train_epoch_times)
            jax.debug.print("")
            print(">>> TRAINING SUMMARY <<<")
            print("===========================================================")
            print(f"total time       :  {total_duration:.4f}")
            print(f"avg train epoch  :  {avg_train:.4f} +/- {std_train:.4f}")
            print("===========================================================")
            print("")

        if self.logger is not None:
            self.logger.flush()
            self.logger.close()
