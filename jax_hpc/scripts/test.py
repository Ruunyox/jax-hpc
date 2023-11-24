import jax_hpc.nn as nn
import optax
import tensorflow_datasets as tfds
import tensorflow as tf
from typing import Tuple, Iterator
import numpy as np
from jax_hpc.nn.training import *

jax.debug.print("Creating model...")
model_init_args = {
    "out_dim": 10,
    "activation": jax.nn.relu,
    "hidden_layers": [512, 256],
}
model = nn.models.FullyConnectedClassifier(**model_init_args)
loss_fn = nn.training.image_cat_cross_entropy
optimizer = optax.adam(0.001)
jax.debug.print("Done...")

jax.debug.print("Creating trainer...")
trainer = nn.training.ClassifierTrainer(model, optimizer, loss_fn)
jax.debug.print("Done...")

jax.debug.print("Initializing datasets...")
dataset, info = tfds.load(
    "fashion_mnist",
    batch_size=512,
    split=["train[0%:80%]", "train[80%:100%]"],
    with_info=True,
)

train_dataset = dataset[0]
val_dataset = dataset[1]
dummy_data = list(train_dataset.take(1))[0]["image"].numpy()

jax.debug.print("Initializing Parameters/Optimizer...")
trainer.set_initial_state(dummy_data)
jax.debug.print("Done...")
trainer.train(train_dataset, val_dataset, num_epochs=300)
