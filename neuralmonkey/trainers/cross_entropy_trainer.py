from typing import Any, List

import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.logging import warn
from neuralmonkey.trainers.generic_trainer import GenericTrainer
from neuralmonkey.trainers.objective import (
    Objective, CostObjective, ObjectiveWeight)


# for compatibility reasons
def xent_objective(decoder, weight=None) -> Objective:
    """Get XENT objective from decoder with cost."""
    warn("Using deprecated xent_objective function. Use the CostObjective "
         "class directly.")
    return CostObjective(decoder, weight)


# pylint: disable=too-many-arguments
class CrossEntropyTrainer(GenericTrainer):

    def __init__(self,
                 decoders: List[Any],
                 decoder_weights: List[ObjectiveWeight] = None,
                 l1_weight: float = 0.,
                 l2_weight: float = 0.,
                 clip_norm: float = None,
                 optimizer: tf.train.Optimizer = None,
                 var_scopes: List[str] = None,
                 var_collection: str = None) -> None:
        check_argument_types()

        if decoder_weights is None:
            decoder_weights = [None for _ in decoders]

        if len(decoder_weights) != len(decoders):
            raise ValueError(
                "decoder_weights (length {}) do not match decoders (length {})"
                .format(len(decoder_weights), len(decoders)))

        objectives = [CostObjective(dec, w)
                      for dec, w in zip(decoders, decoder_weights)]

        GenericTrainer.__init__(
            self,
            objectives=objectives,
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            clip_norm=clip_norm,
            optimizer=optimizer,
            var_scopes=var_scopes,
            var_collection=var_collection)
