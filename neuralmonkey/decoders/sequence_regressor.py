from typing import Callable, List

import tensorflow as tf
import numpy as np

from typeguard import check_argument_types
from neuralmonkey.nn.projection import multilayer_projection
from neuralmonkey.dataset import Dataset
from neuralmonkey.model.model_part import ModelPart, FeedDict, InitializerSpecs
from neuralmonkey.model.stateful import Stateful
from neuralmonkey.decorators import tensor


class SequenceRegressor(ModelPart):
    """A simple MLP regression over encoders.

    The API pretends it is an RNN decoder which always generates a sequence of
    length exactly one.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoders: List[Stateful],
                 data_id: str,
                 layers: List[int] = None,
                 activation_fn: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
                 dropout_keep_prob: float = 1.0,
                 dimension: int = 1,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()

        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint,
                           initializers)

        self.encoders = encoders
        self.data_id = data_id
        self.max_output_len = 1
        self.dimension = dimension

        self._layers = layers
        self._activation_fn = activation_fn
        self._dropout_keep_prob = dropout_keep_prob

        with self.use_scope():
            self.train_inputs = tf.placeholder(
                tf.float32, shape=[None, self.dimension], name="targets")
            self.loss_valid_indices = tf.placeholder(
                tf.int32, [None], name="loss_valid_indices")

        tf.summary.scalar(
            "val_optimization_cost", self.cost,
            collections=["summary_val"])
        tf.summary.scalar(
            "train_optimization_cost",
            self.cost, collections=["summary_train"])
    # pylint: enable=too-many-arguments

    @tensor
    def _mlp_input(self):
        return tf.concat([enc.output for enc in self.encoders], 1)

    @tensor
    def _mlp_output(self):
        return multilayer_projection(
            self._mlp_input, self._layers, self.train_mode,
            self._activation_fn, self._dropout_keep_prob)

    @tensor
    def predictions(self):
        return tf.layers.dense(
            self._mlp_output, self.dimension, name="output_projection")

    @tensor
    def cost(self):
        # TODO handle loss mask
        return tf.reduce_mean(tf.square(
            self.predictions - tf.expand_dims(self.train_inputs, 1)))

    @property
    def train_loss(self):
        return self.cost

    @property
    def runtime_loss(self):
        return self.cost

    @property
    def decoded(self):
        return self.predictions

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = ModelPart.feed_dict(self, dataset, train)
        fd[self.loss_valid_indices] = list(range(len(dataset)))

        targets = dataset.maybe_get_series(self.data_id)
        if targets is not None:
            fd[self.train_inputs] = targets

            # TODO co kdyz chci aby tam byly nuly?
            fd[self.loss_valid_indices] = np.squeeze(np.argwhere(
                np.any(np.array(targets) != 0, axis=1)))

        return fd
