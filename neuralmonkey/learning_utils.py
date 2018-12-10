# pylint: disable=too-many-lines
# TODO de-clutter this file!

from argparse import Namespace
import time
# pylint: disable=unused-import
from typing import (Any, Callable, Dict, List, Tuple, Optional, Union,
                    Iterable, Iterator, Set)
# pylint: enable=unused-import

import numpy as np
import tensorflow as tf
from termcolor import colored

from neuralmonkey.logging import log, log_print, warn
from neuralmonkey.dataset import Dataset
from neuralmonkey.tf_manager import TensorFlowManager
from neuralmonkey.model.feedable import Feedable
from neuralmonkey.runners.base_runner import (
    BaseRunner, ExecutionResult, FeedDict, GraphExecutor)
from neuralmonkey.runners.dataset_runner import DatasetRunner
from neuralmonkey.trainers.generic_trainer import GenericTrainer
from neuralmonkey.trainers.multitask_trainer import MultitaskTrainer
from neuralmonkey.trainers.delayed_update_trainer import DelayedUpdateTrainer
from neuralmonkey.training_profiler import TrainingProfiler

# pylint: disable=invalid-name
Evaluation = Dict[str, float]
SeriesName = str
EvalConfiguration = List[Union[Tuple[SeriesName, Any],
                               Tuple[SeriesName, SeriesName, Any]]]
Postprocess = Optional[List[Tuple[SeriesName, Callable]]]
Trainer = Union[GenericTrainer, MultitaskTrainer, DelayedUpdateTrainer]
# pylint: enable=invalid-name


# pylint: disable=too-many-nested-blocks,too-many-locals
# pylint: disable=too-many-branches,too-many-statements,too-many-arguments
def training_loop(cfg: Namespace) -> None:
    """Execute the training loop for given graph and data.

    Arguments:
        cfg: Experiment configuration namespace.
    """
    _check_series_collisions(cfg.runners, cfg.postprocess)
    _log_model_variables(cfg.trainers)
    _initialize_model(cfg.tf_manager, cfg.initial_variables,
                      cfg.runners + cfg.trainers)

    log("Initializing TensorBoard summary writer.")
    tb_writer = tf.summary.FileWriter(cfg.output,
                                      cfg.tf_manager.sessions[0].graph)
    log("TensorBoard writer initialized.")

    runtime_feedables = set.union(*[ex.feedables for ex in cfg.runners])
    runtime_feedables |= cfg.dataset_runner.feedables
    train_feedables = set.union(*[ex.feedables for ex in cfg.trainers])

    log("Starting training")
    profiler = TrainingProfiler()
    profiler.training_start()

    step = 0
    seen_instances = 0
    last_seen_instances = 0
    interrupt = None

    sess = cfg.tf_manager.sessions[0]

    from neuralmonkey.experiment import Experiment
    exp = Experiment.get_current()

    train_handle = exp.train_handle
    val_handles = exp.val_handles
    test_handles = exp.test_handles
    handle = exp.dataset_handle

    try:
        for epoch_n in range(1, cfg.epochs + 1):

            cfg.tf_manager.init_training()

            # TODO(tf-data) skip train_start_offset
            batch_n = 0

            log_print("")
            log("Epoch {} begins".format(epoch_n), color="red")
            profiler.epoch_start()

            while True:
                try:
                    batch_n += 1
                    step += 1

                    if cfg.log_timer(step, profiler.last_log_time):

                        f_batch = prefetch_dataset(
                            cfg.tf_manager, {handle: train_handle},
                            cfg.dataset_runner)

                        # se zapnutym list zipem v dataset runneru mi to sem
                        # hodí Dict[str:series, List:batch[Tuple:factor[data]]]
                        # ale dataset chce
                        # Dict[str:series, Tuple:factor[List:batch]]

                        bfd = {}
                        for s_id in f_batch.outputs:
                            bfd[s_id] = list(map(list, zip(*f_batch.outputs[s_id])))[0]

                        batch_feed_dict = dict(zip(
                            tf.contrib.framework.nest.flatten(cfg.dataset_runner.dataset),
                            tf.contrib.framework.nest.flatten_up_to(cfg.dataset_runner.dataset, bfd)))

                        exec_result, _ = run_batch(
                            cfg.tf_manager,
                            batch_feed_dict,
                            cfg.runners, cfg.dataset_runner,
                            runtime_feedables,
                            cfg.postprocess,
                            compute_losses=True)

                        trainer_result = cfg.tf_manager.execute(
                            batch_feed_dict, train_feedables, cfg.trainers,
                            train=True, summaries=True)
                        seen_instances += trainer_result[0].size

                        exec_result, _ = run_batch(
                            cfg.tf_manager,
                            batch_feed_dict,
                            cfg.runners, cfg.dataset_runner,
                            runtime_feedables,
                            cfg.postprocess,
                            compute_losses=True)

                        nest = tf.contrib.framework.nest
                        def normalize(arr):
                            if np.issubdtype(arr.dtype, np.number):
                                return arr
                            else:
                                return nest.map_structure(
                                    tf.compat.as_text, arr.tolist())

                        exec_result = exec_result._replace(outputs=nest.map_structure(normalize, exec_result.outputs))
                        input_data = nest.map_structure(normalize, f_batch.outputs)

                        train_evaluation = evaluation(
                            cfg.evaluation, input_data, cfg.runners, exec_result)

                        _log_continuous_evaluation(
                            tb_writer, cfg.main_metric, train_evaluation,
                            seen_instances, epoch_n, cfg.epochs, trainer_result,
                            train=True)

                        profiler.log_done()
                    else:
                        res = cfg.tf_manager.execute(
                            {handle: train_handle},
                            train_feedables, cfg.trainers, train=True,
                            summaries=False)
                        seen_instances += res[0].size

                    if cfg.val_timer(step, profiler.last_val_time):
                        log_print("")
                        profiler.validation_start()
                        val_examples = 0

                        cfg.tf_manager.init_validation()

                        for val_id, valhand in enumerate(val_handles):

                            val_result, f_valset = run_on_dataset(
                                cfg.tf_manager,
                                {handle: valhand},
                                cfg.runners, cfg.dataset_runner,
                                runtime_feedables,
                                cfg.postprocess, write_out=False,
                                compute_losses=True)

                            val_examples += val_result.size

                            valheader = ("Validation (epoch {}, batch number {}):"
                                         .format(epoch_n, batch_n))
                            log(valheader, color="blue")
                            _print_examples(
                                f_valset.outputs, val_result.outputs,
                                f_valset.size,
                                cfg.val_preview_input_series,
                                cfg.val_preview_output_series,
                                cfg.val_preview_num_examples)
                            log_print("")
                            log(valheader, color="blue")

                            val_evaluation = evaluation(
                                cfg.evaluation, f_valset.outputs, cfg.runners,
                                val_result)

                            # The last validation set is selected to be the main
                            if val_id == len(cfg.val_datasets) - 1:
                                this_score = val_evaluation[cfg.main_metric]
                                cfg.tf_manager.validation_hook(this_score, epoch_n,
                                                           batch_n)

                                if this_score == cfg.tf_manager.best_score:
                                    best_score_str = colored(
                                        "{:.4g}".format(cfg.tf_manager.best_score),
                                        attrs=["bold"])

                                    # store also graph parts
                                    rnrs = cfg.runners + cfg.trainers  # type: ignore
                                    # TODO: refactor cfg.trainers/cfg.runners so that they
                                    # have the same API predecessor
                                    parameterizeds = set.union(
                                        *[rnr.parameterizeds
                                          for rnr in rnrs])
                                    for coder in parameterizeds:
                                        for session in cfg.tf_manager.sessions:
                                            coder.save(session)
                                else:
                                    best_score_str = "{:.4g}".format(
                                        cfg.tf_manager.best_score)

                                log("best {} on validation: {} (in epoch {}, "
                                    "after batch number {})"
                                    .format(cfg.main_metric, best_score_str,
                                            cfg.tf_manager.best_score_epoch,
                                            cfg.tf_manager.best_score_batch),
                                    color="blue")

                            v_name = "val_{}".format(val_id) if len(
                                cfg.val_datasets) > 1 else None
                            _log_continuous_evaluation(
                                tb_writer, cfg.main_metric, val_evaluation,
                                seen_instances, epoch_n, cfg.epochs, [val_result],
                                train=False, dataset_name=v_name)

                        profiler.validation_done()
                        profiler.log_after_validation(
                            val_examples, seen_instances - last_seen_instances)
                        last_seen_instances = seen_instances

                        log_print("")

                except tf.errors.OutOfRangeError:
                    break

    except KeyboardInterrupt as ex:
        interrupt = ex

    log("Training finished. Maximum {} on validation data: {:.4g}, epoch {}"
        .format(cfg.main_metric, cfg.tf_manager.best_score,
                cfg.tf_manager.best_score_epoch))

    if cfg.test_datasets:
        cfg.tf_manager.restore_best_vars()
        cfg.tf_manager.init_testing()

        for test_id, testhand in enumerate(test_handles):
            test_results, test_outputs, f_testset = run_on_dataset(
                cfg.tf_manager,
                {handle: testhand},
                cfg.runners, cfg.dataset_runner,
                runtime_feedables,
                cfg.postprocess,
                write_out=True,
                compute_losses=True)
            # ensure test outputs are iterable more than once
            test_outputs = {k: list(v) for k, v in test_outputs.items()}
            eval_result = evaluation(cfg.evaluation, f_testset, cfg.runners,
                                     test_results, test_outputs)
            print_final_evaluation(eval_result, "test_{}".format(test_id))

    if interrupt is not None:
        raise interrupt  # pylint: disable=raising-bad-type


def _log_model_variables(trainers: List[Trainer]) -> None:

    var_list = list(set().union(*[t.var_list for t in trainers])) \
               # type: List[tf.Variable]

    trainable_vars = tf.trainable_variables()
    if not var_list:
        var_list = trainable_vars

    assert var_list is not None
    fixed_vars = [var for var in trainable_vars if var not in var_list]

    total_params = 0

    logstr = "The model has {} trainable variables{}:\n\n".format(
        len(trainable_vars),
        " ({} {})".format(len(fixed_vars), colored("fixed", on_color="on_red"))
        if fixed_vars else "")

    logstr += colored(
        "{: ^80}{: ^20}{: ^10}\n".format("Variable name", "Shape", "Size"),
        color="yellow", attrs=["bold"])

    for var in trainable_vars:

        shape = var.get_shape().as_list()
        params_in_var = int(np.prod(shape))
        total_params += params_in_var

        name = var.name
        if var not in var_list:
            name = colored(name, on_color="on_red")
        # Pad and compensate for control characters:
        name = name.ljust(80 + (len(name) - len(var.name)))
        log_entry = "{}{: <20}{: >10}".format(name, str(shape), params_in_var)
        logstr += "\n{}".format(log_entry)

    logstr += "\n"

    log(logstr)
    log("Total number of all parameters: {}".format(total_params))


def _initialize_model(tf_manager: TensorFlowManager,
                     initial_variables: Optional[List[str]],
                     executables: List[GraphExecutor]):

    if initial_variables is None:
        # Assume we don't look at coder checkpoints when global
        # initial variables are supplied
        tf_manager.initialize_model_parts(executables)
    else:
        try:
            tf_manager.restore(initial_variables)
        except tf.errors.NotFoundError:
            warn("Some variables were not found in checkpoint.)")


def _check_series_collisions(runners: List[BaseRunner],
                             postprocess: Postprocess) -> None:
    """Check if output series names do not collide."""
    runners_outputs = set()  # type: Set[str]
    for runner in runners:
        series = runner.output_series
        if series in runners_outputs:
            raise Exception(("Output series '{}' is multiple times among the "
                             "runners' outputs.").format(series))
        else:
            runners_outputs.add(series)
    if postprocess is not None:
        for series, _ in postprocess:
            if series in runners_outputs:
                raise Exception(("Postprocess output series '{}' "
                                 "already exists.").format(series))
            else:
                runners_outputs.add(series)


def prefetch_dataset(tf_manager: TensorFlowManager,
                     data_feed_dict: FeedDict,
                     dataset_runner: DatasetRunner) -> Dict[str, Any]:
    """Use this function for pre-fetching a batch as a feed dictionary to
    be able to evaluate the model on a single batch multiple times. """

    data_result = tf_manager.execute(data_feed_dict, {dataset_runner},
                                     [dataset_runner])

    return data_result[0]


def run_batch(tf_manager: TensorFlowManager,
              data_feed_dict: FeedDict,
              runners: List[BaseRunner],
              dataset_runner: DatasetRunner,
              feedables: Set[Feedable],
              postprocess: Postprocess,
              compute_losses: bool = False) -> Tuple[ExecutionResult]:
    """The goal of this function is to run a batch and merge execution results
    from different graph executors."""

    executors = []  # type: List[GraphExecutor]
    executors.extend(runners)
    executors.append(dataset_runner)

    execution_results = tf_manager.execute(
        data_feed_dict, feedables, executors, compute_losses=compute_losses)

    sizes = set(ex.size for ex in execution_results)
    assert len(sizes) == 1
    processed_examples = next(iter(sizes))

    results = execution_results[:-1]
    dataset = execution_results[-1]

    # Join execution results from different runners
    result_data = {}
    for s_id, data in (
            pair for res in results for pair in res.outputs.items()):

        # for s_id, data in output.items():
        if s_id in result_data:
            raise ValueError("Overwriting output series forbidden.")
        result_data[s_id] = data

    # Run dataset-level postprocessing.
    if postprocess is not None:
        for s_id, postprocessor in postprocess:
            result_data[s_id] = postprocessor(dataset, result_data)

    # Check output series lengths.
    for s_id, data in result_data.items():
        if len(data) != processed_examples:
            warn("Output '{}' has length {}, but input dataset size is {}"
                 .format(s_id, len(data), processed_examples))

    losses = {}
    for loss_dict in [res.losses for res in results]:
        if any(l in losses for l in loss_dict):
            raise ValueError("Overwriting losses forbidden.")
        losses.update(loss_dict)

    return ExecutionResult(result_data, losses, processed_examples,
                           [res.scalar_summaries for res in execution_results],
                           [res.image_summaries for res in execution_results],
                           [res.histogram_summaries for res in execution_results]), dataset


def run_on_dataset(tf_manager: TensorFlowManager,
                   data_feed_dict: FeedDict,
                   runners: List[BaseRunner],
                   dataset_runner: DatasetRunner,
                   feedables: Set[Feedable],
                   postprocess: Postprocess,
                   write_out: bool = False,
                   log_progress: int = 0,
                   compute_losses: bool = False) -> Tuple[
                       List[ExecutionResult],
                       Dict[str, List],
                       Dict[str, List]]:
    """Apply the model on a dataset and optionally write outputs to files.

    This function processes the dataset in batches and optionally prints out
    the execution progress.

    Args:
        tf_manager: TensorFlow manager with initialized sessions.
        runners: A function that runs the code
        dataset_runner: A runner object that fetches the data inputs
        data_feed_dict: Feed dict that provides the dataset.
        evaluators: List of evaluators that are used for the model
            evaluation if the target data are provided.
        postprocess: Dataset-level postprocessors
        write_out: Flag whether the outputs should be printed to a file defined
            in the dataset object.
        log_progress: log progress every X seconds

    Returns:
        Tuple of resulting sentences/numpy arrays, and evaluation results if
        they are available which are dictionary function -> value.

    """
    last_log_time = time.process_time()
    batch_results = []
    batch_inputs = []

    processed_examples = 0

    nest = tf.contrib.framework.nest
    def normalize(arr):
        if np.issubdtype(arr.dtype, np.number):
            return arr
        else:
            return nest.map_structure(
                tf.compat.as_text, arr.tolist())

    while True:
        try:
            if 0 < log_progress < time.process_time() - last_log_time:
                log("Processed {} examples.".format(processed_examples))
                last_log_time = time.process_time()

            result, dataset = run_batch(tf_manager, data_feed_dict, runners,
                                        dataset_runner, feedables, postprocess,
                                        compute_losses)

            batch_results.append(result._replace(
                outputs=nest.map_structure(normalize, result.outputs)))
            batch_inputs.append(dataset._replace(
                outputs=nest.map_structure(normalize, dataset.outputs)))

            processed_examples += result.size
        except tf.errors.OutOfRangeError:
            break

    # Join execution results from different batches. Note that the arrays can
    # differ in both batch and time dimensions.
    joined_result = join_execution_results(batch_results)
    joined_inputs = join_execution_results(batch_inputs, dataset_runner.dataset)

    #for res in batch_results]
    # output_results = joined_result
    # dataset = joined_results[-1].outputs

    # fetched_input = {
    #     k: [dic[k] for dic in input_transposed] for k in input_transposed[0]}

    # TODO(tf-data) nested structures
    # fetched_input_lengths = {s: len(fetched_input[s]) for s in fetched_input}

    # if len(set(fetched_input_lengths.values())) != 1:
    #     warn("Fetched input dataset series are not of the same length: {}"
    #          .format(str(fetched_input_lengths)))

    # # TODO(tf-data) this does not work when series are nested
    # dataset_len = fetched_input_lengths[next(iter(fetched_input_lengths))]

    # Join execution results from different runners

    # result_data = {}
    # for s_id, data in (pair for res in output_results
    #                    for pair in res.outputs.items()):

    #     # for s_id, data in output.items():
    #     if s_id in result_data:
    #         raise ValueError("Overwriting output series forbidden.")
    #     result_data[s_id] = data

    # # Run dataset-level postprocessing.
    # if postprocess is not None:
    #     for s_id, postprocessor in postprocess:
    #         result_data[s_id] = postprocessor(dataset, result_data)

    # # Check output series lengths.
    # for s_id, data in result_data.items():
    #     if len(data) != processed_examples:
    #         warn("Output '{}' has length {}, but input dataset size is {}"
    #              .format(s_id, len(data), processed_examples))

    # POZOR TOHLE SE NESMI SMAZAT Z run_on_dataset !!! (anebo se to dá dál)
    # TODO(tf-data)
    # if write_out and dataset.outputs is not None:
    #     for series_id, data in result_data.items():
    #         if series_id in dataset.outputs:
    #             path, writer = dataset.outputs[series_id]
    #             writer(path, data)
    #         else:
    #             log("There is no file for output series '{}' in dataset: '{}'"
    #                 .format(series_id, dataset.name), color="red")
    # elif write_out:
    #     log("Dataset does not have any outputs, nothing to write out.",
    #         color="red")

    return joined_result, joined_inputs


# TODO(tf-data) add unit tests!
def join_execution_results(
        execution_results: List[ExecutionResult],
        output_structure: Any = None) -> ExecutionResult:
    """Aggregate execution results into one."""
    losses_sum = {loss: 0. for loss in execution_results[0].losses}

    def join(*args):
        joined = []

        for arg in args:
            if isinstance(arg, np.ndarray):
                joined += np.split(arg, arg.shape[0])
            elif isinstance(arg, list):
                joined += arg
            else:
                raise NotImplementedError("Unsuported output sequence type")
        return joined

    if output_structure is None:
        # Assume dictionary of elements that should be concatenated
        output_structure = {key: 0 for key in execution_results[0].outputs}

    joined_outputs = tf.contrib.framework.nest.map_structure_up_to(
        output_structure, join, *(res.outputs for res in execution_results))

    for result in execution_results:
        for l_id, loss in result.losses.items():
            losses_sum[l_id] += loss * result.size

        # TODO aggregate TensorBoard summaries

    # TODO(tf-data) figure out why there were the following lines:
    # if outputs and isinstance(outputs[0], np.ndarray):
    #    outputs = np.array(outputs)

    total_size = sum(ex.size for ex in execution_results)
    losses = {l_id: loss / total_size for l_id, loss in losses_sum.items()}

    return ExecutionResult(joined_outputs, losses_sum, total_size,
                           execution_results[0].scalar_summaries,
                           execution_results[0].histogram_summaries,
                           execution_results[0].image_summaries)


def evaluation(evaluators, batch, runners, execution_result):
    """Evaluate the model outputs.

    Args:
        evaluators: List of tuples of series and evaluation functions.
        batch: Batch of data against which the evaluation is done.
        runners: List of runners (contains series ids and loss names).
        execution_results: Execution results that include the loss values.
        result_data: Dictionary from series names to list of outputs.

    Returns:
        Dictionary of evaluation names and their values which includes the
        metrics applied on respective series loss and loss values from the run.
    """
    # losses
    eval_result = execution_result.losses

    # evaluation metrics
    for hyp_id, ref_id, evaluator in evaluators:
        if ref_id not in batch or hyp_id not in execution_result.outputs:
            continue

        references = [tup[0] for tup in batch[ref_id]]
        hypotheses = [tup[0] for tup in execution_result.outputs[hyp_id]]

        eval_key = "{}/{}".format(hyp_id, evaluator.name)
        eval_result[eval_key] = evaluator(hypotheses, references)

    return eval_result


def _log_continuous_evaluation(tb_writer: tf.summary.FileWriter,
                               main_metric: str,
                               eval_result: Evaluation,
                               seen_instances: int,
                               epoch: int,
                               max_epochs: int,
                               execution_results: ExecutionResult,
                               train: bool = False,
                               dataset_name: str = None) -> None:
    """Log the evaluation results and the TensorBoard summaries."""

    color, prefix = ("yellow", "train") if train else ("blue", "val")

    if dataset_name is not None:
        prefix += "_" + dataset_name

    eval_string = _format_evaluation_line(eval_result, main_metric)
    eval_string = "Epoch {}/{}  Instances {}  {}".format(epoch, max_epochs,
                                                         seen_instances,
                                                         eval_string)
    log(eval_string, color=color)

    if not tb_writer:
        return

    for summaries in [summ for res in execution_results
                      for summ in (res.scalar_summaries,
                                   res.histogram_summaries,
                                   res.image_summaries)]:
        if summaries is None:
            continue

        if isinstance(summaries, list):
            for summary in summaries:
                if summary is not None:
                    tb_writer.add_summary(summary, seen_instances)
        else:
            tb_writer.add_summary(summaries, seen_instances)

        external_str = \
            tf.Summary(value=[tf.Summary.Value(tag=prefix + "_" + name,
                                               simple_value=value)
                              for name, value in eval_result.items()])
        tb_writer.add_summary(external_str, seen_instances)


def _format_evaluation_line(evaluation_res: Evaluation,
                            main_metric: str) -> str:
    """Format the evaluation metric for stdout with last one bold."""
    eval_string = "    ".join("{}: {:.4g}".format(name, value)
                              for name, value in evaluation_res.items()
                              if name != main_metric)

    eval_string += colored(
        "    {}: {:.4g}".format(main_metric,
                                evaluation_res[main_metric]),
        attrs=["bold"])

    return eval_string


def print_final_evaluation(eval_result: Evaluation, name: str = None) -> None:
    """Print final evaluation from a test dataset."""
    line_len = 22

    if name is not None:
        log("Model evaluated on '{}'".format(name))

    for eval_name, value in eval_result.items():
        space = "".join([" " for _ in range(line_len - len(eval_name))])
        log("... {}:{} {:.4g}".format(eval_name, space, value))

    log_print("")


def _data_item_to_str(item: Tuple) -> str:

    if len(item) == 1:
        return _data_item_to_str2(item[0])

    items = [_data_item_to_str2(i) for i in item]
    return "({})".format(items)

def _data_item_to_str2(item: Any) -> str:
    if isinstance(item, list):
        return " ".join([_data_item_to_str2(i) for i in item])

    if isinstance(item, dict):
        return "{\n      " + "\n      ".join(
            ["{}: {}".format(_data_item_to_str2(key), _data_item_to_str2(val))
             for key, val in item.items()]) + "\n    }"

    if isinstance(item, np.ndarray) and len(item.shape) > 1:
        return "[numpy tensor, shape {}]".format(item.shape)

    return str(item)


def _print_examples(dataset: Dict[str, List],
                    outputs: Dict[str, List],
                    dataset_size: int,
                    val_preview_input_series: Optional[List[str]] = None,
                    val_preview_output_series: Optional[List[str]] = None,
                    num_examples=15) -> None:
    """Print examples of the model output.

    Arguments:
        dataset: The dataset from which to take examples
        outputs: A mapping from the output series ID to the list of its
            contents
        val_preview_input_series: An optional list of input series to include
            in the preview. An input series is a data series that is present in
            the dataset. It can be either a target series (one that is also
            present in the outputs, i.e. reference), or a source series (one
            that is not among the outputs). In the validation preview, source
            input series and preprocessed target series are yellow and target
            (reference) series are red. If None, all series are written.
        val_preview_output_series: An optional list of output series to include
            in the preview. An output series is a data series that is present
            among the outputs. In the preview, magenta is used as the font
            color for output series
    """
    log_print(colored("Examples:", attrs=["bold"]))

    source_series_names = [s for s in dataset if s not in outputs]
    target_series_names = [s for s in dataset if s in outputs]
    output_series_names = list(outputs.keys())

    assert outputs

    if val_preview_input_series is not None:
        target_series_names = [s_id for s_id in target_series_names
                               if s_id in val_preview_input_series]
        source_series_names = [s_id for s_id in source_series_names
                               if s_id in val_preview_input_series]

    if val_preview_output_series is not None:
        output_series_names = [s for s in output_series_names
                               if s in val_preview_output_series]

    # for further indexing we need to make sure, all relevant
    # dataset series are lists
    target_series = {s_id: dataset[s_id] for s_id in target_series_names}
    source_series = {s_id: dataset[s_id] for s_id in source_series_names}

    for i in range(min(num_examples, dataset_size)):
        log_print(colored("  [{}]".format(i + 1), color="magenta",
                          attrs=["bold"]))

        def print_line(prefix, color, content):
            colored_prefix = colored(prefix, color=color)
            formatted = _data_item_to_str(content)
            log_print("  {}: {}".format(colored_prefix, formatted))

        # Input source series = yellow
        for series_id, data in sorted(source_series.items(),
                                      key=lambda x: x[0]):
            print_line(series_id, "yellow", data[i])

        # Output series = magenta
        for series_id in sorted(output_series_names):
            data = list(outputs[series_id])
            model_output = data[i]
            print_line(series_id, "magenta", model_output)

        # Input target series (a.k.a. references) = red
        for series_id in sorted(target_series_names):
            data = outputs[series_id]
            desired_output = target_series[series_id][i]
            print_line(series_id + " (ref)", "red", desired_output)

        log_print("")


def _skip_lines(start_offset: int,
                batches: Iterator[Dataset]) -> None:
    """Skip training instances from the beginning.

    Arguments:
        start_offset: How many training instances to skip (minimum)
        batches: Iterator over batches to skip
    """
    log("Skipping first {} instances in the dataset".format(start_offset))

    skipped_instances = 0
    while skipped_instances < start_offset:
        try:
            skipped_instances += len(next(batches))  # type: ignore
        except StopIteration:
            raise ValueError("Trying to skip more instances than "
                             "the size of the dataset")

    if skipped_instances > 0:
        log("Skipped {} instances".format(skipped_instances))
