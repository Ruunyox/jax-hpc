import argparse
import tensorflow_datasets as tfds
from jsonargparse import CLI, ArgumentParser, ActionConfigFile
from ..nn.models import *
from typing import List, Optional, Any, Union, Iterator, Tuple
import os
import tensorflow as tf
import numpy as np


def dataset_to_iterator(
    ds: tf.data.Dataset, batch_size=int
) -> Tuple[Iterator, np.ndarray]:
    ds.cache()
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x: Batch(**x))
    return iter(tfds.as_numpy(ds)), ds.take(1)


def main():
    parser = ArgumentParser()
    parser.add_function_arguments(tfds.load, "dataset")
    parser.add_argument("--config", action=ActionConfigFile)

    cfg = parser.parse_args()

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

    train_dataset, dummy_input = dataset_to_iterator(train_dataset, cfg.batch_size)
    val_dataset, _ = dataset_to_iterator(val_dataset, cfg.batch_size)

    cfg.model_wrapper.model.fit(
        x=train_dataset, validation_data=val_dataset, **fit_opts
    )


if __name__ == "__main__":
    main()
