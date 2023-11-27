import argparse
from jsonargparse import CLI, ArgumentParser, ActionConfigFile
from ..nn.models import *
from typing import List, Optional, Any, Union, Iterator, Tuple, Iterator
import os
import tensorflow as tf
import numpy as np
import jax_hpc.nn as nn
import optax
import tensorflow_datasets as tfds
import numpy as np
from jax_hpc.nn.training import *

loss_functions = {"image_cat_cross_entropy": image_cat_cross_entropy}
platforms = tuple(["cpu", "gpu", "tpu"])
optimizers = {"optax.adam": optax.adam}


def main():
    parser = ArgumentParser()
    parser.add_argument("--platform", type=str, default="cpu")
    parser.add_argument("--optimizer", type=Dict[str, Any])
    parser.add_argument("--loss_function", type=str)
    parser.add_class_arguments(ModelWrapper, "model")
    parser.add_class_arguments(FitWrapper, "fit")
    parser.add_function_arguments(tfds.load, "dataset")
    parser.add_argument("--config", action=ActionConfigFile)

    cfg = parser.parse_args()

    # Platform
    assert cfg.platform in platforms
    jax.config.update("jax_platform_name", cfg.platform)

    # Optimizer
    opt_name = cfg.optimizer.pop("name")
    assert opt_name in optimizers
    optimizer = optimizers[opt_name](**cfg.optimizer)

    # Loss function
    assert cfg.loss_function in loss_functions
    loss_fn = loss_functions[cfg.loss_function]

    cfg = parser.instantiate_classes(cfg)

    # Dataset

    # disable auto cache:
    rc = tfds.ReadConfig(try_autocache=False)
    cfg.dataset["read_config"] = rc
    dataset = tfds.load(**cfg.dataset)
    if isinstance(dataset, list):
        if len(dataset) not in [1, 2]:
            raise RuntimeError("Dataset splitting only supports two splits")
        else:
            if len(dataset) == 1:
                train_dataset = dataset[0]
                val_dataset = None
            if len(dataset) == 2:
                train_dataset = dataset[0]
                val_dataset = dataset[1]
    elif isinstance(dataset, dict):
        train_dataset = dataset["train"]
        val_dataset = None

    trainer = nn.training.ClassifierTrainer(cfg.model.model, optimizer, loss_fn)
    dummy_data = list(train_dataset.take(1))[0]["image"].numpy()
    trainer.set_initial_state(dummy_data)
    trainer.train(train_dataset, val_dataset, **cfg.fit._asdict())


if __name__ == "__main__":
    main()
