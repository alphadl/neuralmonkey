from abc import abstractmethod, abstractproperty
from typing import (Any, Dict, Tuple, List, NamedTuple, Union, Set, TypeVar,
                    Generic, Optional)
import numpy as np
import tensorflow as tf

from neuralmonkey.logging import notice
from neuralmonkey.model.model_part import GenericModelPart
from neuralmonkey.model.feedable import Feedable
from neuralmonkey.model.parameterized import Parameterized

# pylint: disable=invalid-name
FeedDict = Dict[tf.Tensor, Union[int, float, np.ndarray]]
NextExecute = Tuple[Union[Dict, List], List[FeedDict]]
MP = TypeVar("MP", bound=GenericModelPart)
Executor = TypeVar("Executor", bound="GraphExecutor")
Runner = TypeVar("Runner", bound="BaseRunner")
# pylint: enable=invalid-name


class ExecutionResult(NamedTuple(
        "ExecutionResult",
        [("outputs", Any),
         ("losses", List[float]),
         ("scalar_summaries", tf.Summary),
         ("histogram_summaries", tf.Summary),
         ("image_summaries", tf.Summary)])):
    """A data structure that represents a result of a graph execution.

    The goal of each runner is to populate this structure and set it as its
    ``self._result``.

    Attributes:
        outputs: A nested structure of batched outputs of the runner.
        losses: A (possibly empty) list of loss values computed during the run.
        scalar_summaries: A TensorFlow summary object with scalar values.
        histogram_summaries: A TensorFlow summary object with histograms.
        image_summaries: A TensorFlow summary object with images.
    """


class GraphExecutor(GenericModelPart):

    class Executable(Generic[Executor]):

        def __init__(self,
                     executor: Executor,
                     compute_losses: bool,
                     summaries: bool,
                     num_sessions: int) -> None:
            self._executor = executor
            self.compute_losses = compute_losses
            self.summaries = summaries
            self.num_sessions = num_sessions

            self._result = None  # type: Optional[ExecutionResult]

        def set_result(self, outputs: List[Any], losses: List[float],
                       scalar_summaries: tf.Summary,
                       histogram_summaries: tf.Summary,
                       image_summaries: tf.Summary) -> None:
            self._result = ExecutionResult(
                outputs, losses, scalar_summaries, histogram_summaries,
                image_summaries)

        @property
        def result(self) -> Optional[ExecutionResult]:
            return self._result

        @property
        def executor(self) -> Executor:
            return self._executor

        def next_to_execute(self) -> NextExecute:
            """Get the tensors and additional feed dicts for execution."""
            return self.executor.fetches, [{}]

        @abstractmethod
        def collect_results(self, results: List[Dict]) -> None:
            return None

    def __init__(self,
                 dependencies: Set[GenericModelPart]) -> None:
        self._dependencies = dependencies
        self._feedables, self._parameterizeds = self.get_dependencies()

    def get_executable(self,
                       compute_losses: bool,
                       summaries: bool,
                       num_sessions: int) -> "GraphExecutor.Executable":
        # Since the executable is always subclassed, we can instantiate it
        return self.Executable(  # type: ignore
            self, compute_losses, summaries, num_sessions)

    @abstractproperty
    def fetches(self) -> Dict[str, tf.Tensor]:
        raise NotImplementedError()

    @property
    def dependencies(self) -> List[str]:
        return ["_dependencies"]

    @property
    def feedables(self) -> Set[Feedable]:
        return self._feedables

    @property
    def parameterizeds(self) -> Set[Parameterized]:
        return self._parameterizeds


class BaseRunner(GraphExecutor, Generic[MP]):

    # pylint: disable=too-few-public-methods
    # Pylint issue here: https://github.com/PyCQA/pylint/issues/2607
    class Executable(GraphExecutor.Executable[Runner]):

        def next_to_execute(self) -> NextExecute:
            fetches = self.executor.fetches

            if not self.compute_losses:
                for loss in self.executor.loss_names:
                    fetches[loss] = tf.zeros([])

            return fetches, [{}]
    # pylint: enable=too-few-public-methods

    def __init__(self,
                 output_series: str,
                 decoder: MP) -> None:
        GraphExecutor.__init__(self, {decoder})
        self.output_series = output_series
        self.decoder = decoder

        if not hasattr(decoder, "data_id"):
            notice("Top-level decoder {} does not have the 'data_id' attribute"
                   .format(decoder))

    @property
    def decoder_data_id(self) -> Optional[str]:
        return getattr(self.decoder, "data_id", None)

    @property
    def loss_names(self) -> List[str]:
        raise NotImplementedError()


def reduce_execution_results(
        execution_results: List[ExecutionResult]) -> ExecutionResult:
    """Aggregate execution results into one."""
    outputs = []  # type: List[Any]
    losses_sum = [0. for _ in execution_results[0].losses]
    for result in execution_results:
        outputs.extend(result.outputs)
        for i, loss in enumerate(result.losses):
            losses_sum[i] += loss
        # TODO aggregate TensorBoard summaries
    if outputs and isinstance(outputs[0], np.ndarray):
        outputs = np.array(outputs)
    losses = [l / max(len(outputs), 1) for l in losses_sum]
    return ExecutionResult(outputs, losses,
                           execution_results[0].scalar_summaries,
                           execution_results[0].histogram_summaries,
                           execution_results[0].image_summaries)
