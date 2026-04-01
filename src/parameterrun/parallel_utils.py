import contextlib
import inspect
import sys
from collections.abc import Iterable
from datetime import datetime
from itertools import product
from time import time
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import joblib
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def _normalized_time(time_seg: float) -> str:
    """
    Normalize the time to a human-readable format
    """
    if time_seg < 60:
        return f'{time_seg:.2f} s'
    elif time_seg < 3600:
        mins, sec = divmod(time_seg, 60)
        return f'{int(mins)} min {sec:.2f} s'
    else:
        hours, time_seg = divmod(time_seg, 3600)
        mins, sec = divmod(time_seg, 60)
        return f'{int(hours)} h {int(mins)} min {sec:.2f} s'


def _log(message: str, verbose: Optional[bool] = False, hostname: Optional[str] = None) -> None:
    if verbose:
        # Compute timestamp
        time_stamp = datetime.now().strftime("%H:%M:%S, %d/%m/%Y")
        msg = f'{time_stamp}: {message}'
        if hostname is not None:
            msg = f'[{hostname}]: {msg}'

        print(msg, flush=True)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def _parameterrun_joblib(fun: Callable[..., Any], param_names: List[List[str]], param_values: List[List[List]],
                         indices_iterate: list, desc, n_workers: Optional[int] = -1, pbar_bool: bool = True,
                         leave: Optional[bool] = True, **fun_kwargs) -> Tuple[List[List], int]:
    n_total = len(indices_iterate)
    if n_total == 0:
        raise ValueError('No parameter combinations generated. Please provide at least one value per group.')

    result = []
    if n_workers == 1:  # If only one worker, do not use joblib
        pbar = tqdm(indices_iterate, desc=desc, leave=leave, disable=not pbar_bool)
        for index in pbar:
            result.append(fun(**{**_get_iteration(index, param_names, param_values), **fun_kwargs}))
    else:
        with tqdm_joblib(tqdm(indices_iterate, desc=desc, leave=leave, disable=not pbar_bool)) as _:
            result = Parallel(n_jobs=n_workers)(
                delayed(fun)(**{**_get_iteration(index, param_names, param_values), **fun_kwargs}) for index in
                indices_iterate)

    n_output = len(result[0]) if isinstance(result[0], tuple) else 1
    if n_output == 1:
        result = [(result_i,) for result_i in result]

    result_temp = []
    for i in range(n_output):
        result_temp.append([result[j][i] for j in range(n_total)])

    return result_temp, n_output


def _parameterrun_mpi(fun: Callable[..., Any], param_names: List[List[str]], param_values: List[List[List]],
                      groups_iterate: list, desc: Optional[str], pbar_bool: bool = True,
                      verbose: Optional[bool] = False, leave: Optional[bool] = True, **fun_kwargs) -> Tuple[
    List[List], int]:  # pragma: no cover
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_total = len(groups_iterate)

    # Shuffle the indices to avoid bias in the computation
    if rank == 0:
        total_indices = np.arange(n_total)
        np.random.shuffle(total_indices)
    else:
        total_indices = None

    total_indices = comm.bcast(total_indices, root=0)
    indices_compute = np.array_split(total_indices, size)[rank]

    groups_compute = np.array(groups_iterate)[indices_compute]

    pbar = tqdm(groups_compute, desc=desc, leave=leave, disable=not (pbar_bool and rank == 0), file=sys.stdout)

    n_outputs = 0
    result: List[Any] = []

    if 'Open MPI v5.0.6' in MPI.Get_library_version():
        mpi_flush = True
    else:
        mpi_flush = False

    for index in pbar:
        if pbar_bool and rank == 0 and mpi_flush:
            print('\r', flush=True)  # Needed to show the progress bar in some mpi versions
            sys.stdout.write("\033[F")  # Move the cursor up one line

        result_temp = fun(**{**_get_iteration(index, param_names, param_values), **fun_kwargs})

        if not isinstance(result_temp, tuple):
            result_temp = (result_temp,)

        # Check how many parameters are returned
        if len(result) == 0:  # First iteration
            n_outputs = len(result_temp)
            for _ in range(n_outputs):
                result.append([])

        for i, result_temp_i in enumerate(result_temp):
            result[i].append(result_temp_i)

    _log(f'Rank {rank} finished', verbose, hostname=MPI.Get_processor_name())

    # Notify root when each worker is done, and show a progress bar
    if rank == 0:
        progress = 0
        pbar = tqdm(total=size - 1, desc='Waiting for workers', leave=leave, disable=not pbar_bool, file=sys.stdout)

        while progress < size - 1:
            if comm.Iprobe(source=MPI.ANY_SOURCE, tag=0):
                if pbar_bool and mpi_flush:
                    print('\r', flush=True)  # Needed to show the progress bar in some mpi versions
                    sys.stdout.write("\033[F")

                comm.recv(source=MPI.ANY_SOURCE, tag=0)  # Receive a signal from each worker
                progress += 1
                pbar.update(1)

        pbar.close()
    else:
        comm.send(1, dest=0, tag=0)  # Notify root that the work is done

    n_outputs = comm.bcast(n_outputs, root=0)

    if not result:  # For workers that did not compute anything
        result = [[] for _ in range(n_outputs)]

    results_gathered = []
    for i in range(n_outputs):
        _log(f'Gathering results {i}', verbose and rank == 0, hostname=MPI.Get_processor_name())

        result_gathered = comm.gather(result[i], root=0)

        _log('Results gathered', verbose and rank == 0, hostname=MPI.Get_processor_name())

        if rank == 0:
            result_gathered = [item for sublist in result_gathered for item in sublist]  # Flatten the list

            results_unshuffled = [None] * n_total
            for j, index in enumerate(total_indices):
                results_unshuffled[index] = result_gathered[j]

            results_gathered.append(results_unshuffled)

        _log('Results reshaped', verbose and rank == 0, hostname=MPI.Get_processor_name())

    return results_gathered, n_outputs


def _get_iteration(index, param_names, param_values):
    n_groups = len(param_values)
    n_parameters = [len(group) for group in param_values]

    dic_temp = {}
    for group_index in range(n_groups):
        for parameter_index in range(n_parameters[group_index]):
            dic_temp[param_names[group_index][parameter_index]] = (
                param_values[group_index][parameter_index][index[group_index]])

    return dic_temp


def _flatten_param_names(param_names: List[List[str]]) -> List[str]:
    return [name for group in param_names for name in group]


def _to_list_iterable(values: Any, error_message: str) -> List[Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(error_message)
    return list(values)


def _validate_function_arguments(fun: Callable[..., Any], param_names: List[List[str]], fun_kwargs: dict) -> None:
    signature = inspect.signature(fun)
    parameters = signature.parameters
    accepts_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())

    sweep_names = _flatten_param_names(param_names)

    duplicated = sorted({name for name in sweep_names if sweep_names.count(name) > 1})
    if duplicated:
        raise ValueError(f'Duplicate parameter names are not allowed: {duplicated}')

    overlapping_kwargs = sorted(set(sweep_names).intersection(fun_kwargs))
    if overlapping_kwargs:
        raise ValueError(f'Keyword arguments conflict with swept parameters: {overlapping_kwargs}. '
                         f'Remove them from kwargs or param_names.')

    if accepts_var_kwargs:
        return

    known_names = set(parameters)
    unknown_sweep = sorted(name for name in sweep_names if name not in known_names)
    if unknown_sweep:
        raise ValueError(f'Unknown swept parameter names for function {fun.__name__}: {unknown_sweep}')

    unknown_kwargs = sorted(name for name in fun_kwargs if name not in known_names)
    if unknown_kwargs:
        raise ValueError(f'Unknown keyword arguments for function {fun.__name__}: {unknown_kwargs}')


def _format_input(param_names: Union[str, List[str], List[List[str]]], param_values: Any) -> Tuple[
    List[List[str]], List[List[List[Any]]]]:
    # Check the depth of the parameters

    if isinstance(param_names, str):
        if not param_names:
            raise ValueError('Parameter names must be non-empty strings')
        depth = 1
    elif isinstance(param_names, list) and len(param_names) > 0 and all(isinstance(name, str) for name in param_names):
        if any(not name for name in param_names):
            raise ValueError('Parameter names must be non-empty strings')
        depth = 2
    elif isinstance(param_names, list) and len(param_names) > 0 and all(
            isinstance(group, list) for group in param_names):
        if any(len(group) == 0 for group in param_names):
            raise ValueError('Parameter name groups must contain at least one parameter')
        if any(any(not isinstance(name, str) or not name for name in group) for group in param_names):
            raise ValueError('Parameter names must be non-empty strings')
        depth = 3
    else:
        raise ValueError('Unknown input format for the parameters names')

    normalized_param_names: List[List[str]]
    normalized_param_values: List[List[List[Any]]]

    # Correct the input if the depth is not equal to 3
    if depth == 1:
        # Only one parameter is provided
        values = _to_list_iterable(param_values, 'The parameters values must be an iterable of values')
        normalized_param_names = [[cast(str, param_names)]]
        normalized_param_values = [[values]]
    elif depth == 2:
        # Groups with a single parameters are provided
        values = _to_list_iterable(param_values, 'The parameters values must be an iterable of values')
        normalized_param_names = [[param_name_i] for param_name_i in cast(List[str], param_names)]
        normalized_param_values = [
            [_to_list_iterable(param_values_i, 'Each parameter values entry must be an iterable')] for param_values_i in
            values]
    else:
        groups_values = _to_list_iterable(param_values, 'The parameters values must be an iterable of values')
        groups_values = [_to_list_iterable(values_group, 'Each parameter group values must be provided as an iterable')
                         for values_group in groups_values]
        normalized_param_names = cast(List[List[str]], param_names)
        normalized_param_values = [
            [_to_list_iterable(parameter_values, 'Each parameter values entry must be an iterable') for parameter_values
             in values_group] for values_group in groups_values]

    # Check that the number of groups of parameters is the same as the number of groups of values
    if len(normalized_param_names) != len(normalized_param_values):
        raise ValueError('The number of groups of parameters must be the same as the number of groups of values')

    for names_group, values_group in zip(normalized_param_names, normalized_param_values):
        if len(names_group) != len(values_group):
            raise ValueError('Each parameter group must include one values list per parameter')

    # Check that all parameters in the same group have the same length
    for group in normalized_param_values:
        if len(set([len(param) for param in group])) != 1:
            raise ValueError('All parameters in the same group must have the same length')

    all_names = _flatten_param_names(normalized_param_names)
    duplicated = sorted({name for name in all_names if all_names.count(name) > 1})
    if duplicated:
        raise ValueError(f'Duplicate parameter names are not allowed: {duplicated}')

    return normalized_param_names, normalized_param_values


def _reshape_channel_result(channel_values: List[Any], n_values: List[int]) -> np.ndarray:
    array_result = np.array(channel_values)
    if array_result.ndim == 0:
        raise ValueError('Could not reshape scalar output')

    target_shape = tuple(n_values) + tuple(array_result.shape[1:])
    return array_result.reshape(target_shape)


def _convert_result_to_lists(result: Any) -> Any:
    if isinstance(result, np.ndarray):
        return result.tolist()
    if isinstance(result, list):
        return [_convert_result_to_lists(item) for item in result]
    return result


def parameterrun(fun: Callable[..., Any], param_names: Union[str, List[str], List[List[str]]],
                 param_values: Union[Iterable[Any], Iterable[Iterable[Any]], Iterable[Iterable[Iterable[Any]]]],
                 n_workers: Optional[int] = -1, pbar_bool: bool = True, verbose: Optional[bool] = False,
                 reshape: bool = True, result_as_array: bool = True, backend: Optional[str] = None,
                 desc: Optional[str] = None, **kwargs) -> Union[list, np.ndarray, None]:  # noqa: E501
    """
    Run a function with multiple parameters in parallel. To indentify the parameters of interest, user must provide its
    name as is written in the function definition. If more parameters should be pass to the function, they can be
    provided as kwargs. The backend used for the parallelization is chosen automatically between joblib and mpi4py.
    If mpi4py is used, the function must be run with mpirun -n n_workers python script.py. Finally, the result can be
    reshaped as a (hyper)matrix if the reshape parameter is set to True, if the functions return multiple values,
    the result is a list of arrays.

    The parameters can be grouped in different ways, the possible options are:
    - A single parameter: The function is run with a single parameter. In this case, param_names='param_name' and
    param_values=[value1, value2, ...]
    - Multiple parameters: The function is run with multiple parameters. In this case, param_names=['param_name1',
    'param_name2', ...] and param_values=[[value1, value2, ...], [value1, value2, ...], ...]
    - Multiple groups of parameters: The function is run with multiple groups of parameters, running the parameters in
    the same group at the same time. In this case, param_names=[['param_name1', 'param_name2', ...], ['param_name1',
    'param_name2', ...], ...] and param_values=[[[value1, value2, ...], [value1, value2, ...], ...], [[value1, value2,
    ...], [value1, value2, ...], ...], ...]. Note that in this case, the number of values for each parameter in the same
    group must be the same.

    Parameters
    ----------
    fun : callable
        Function to run in parallel.
    param_names : str or list
        Name of the parameters to run in parallel.
    param_values : iterable
        Values of the parameters to run in parallel. Lists, tuples, ranges, generators, and numpy arrays are
        accepted as long as they follow one of the supported input shapes.
    n_workers : int, optional (default=-1)
        Number of workers to use in parallel. If -1, the number of workers is the maximum number of cores available.
    pbar_bool : bool, optional (default=True)
        If True, show a progress bar.
    backend : str, optional (default=None)
        Backend to use for the parallelization. If None, the backend is chosen automatically between joblib and mpi.
    verbose : bool, optional (default=False)
        If True, print information about the parallelization.
    reshape : bool, optional (default=True)
        If True, reshape the result as a (hyper)matrix. Sometimes, due to the nature of the output, the reshape is not
        possible, in this case, the result is not reshaped.
    result_as_array : bool, optional (default=True)
        If True, return numpy arrays when possible (default behavior). If False, convert numpy-array outputs to Python
        lists.
    desc: str, optional (default=None)
        Description of the progress bar. If None, the description denotes the function and parameters names.
    kwargs :
        Additional parameters to pass to the function.

    Returns
    -------
    result : list or np.ndarray
        Result of the function run in parallel. The shape of the (hyper)matrix is (n_values1, n_values2, ...,
         n_valuesN), where n_valuesI is the number of values in the I-th group.
    """

    time_start = time()

    formatted_param_names, formatted_param_values = _format_input(param_names, param_values)
    _validate_function_arguments(fun, formatted_param_names, kwargs)

    # Create the list of dictionaries with the parameters in a nested loop
    n_values = [len(group[0]) for group in formatted_param_values]
    if any(n_value == 0 for n_value in n_values):
        raise ValueError('Parameter groups must contain at least one value.')

    indices = [list(range(n_value)) for n_value in n_values]
    indices_iterate = product(*indices)

    # Choose the backed
    if backend is not None:
        available_backends = ['joblib', 'mpi']
        if backend not in available_backends:
            raise ValueError(f'Unknown backend {backend}. Available backends are {available_backends}')

    size = 1
    rank = 0

    mpi_module = None
    if backend is None or backend == 'mpi':
        try:
            from mpi4py import MPI
            mpi_module = MPI
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
        except ImportError as err:
            if backend == 'mpi':
                raise ImportError('backend="mpi" requires mpi4py to be installed and available.') from err

        if size == 1 and backend is None:
            backend = 'joblib'
        else:
            backend = 'mpi'

    if desc is None:
        param_names_pbar = [parameters_names_i for parameters_names_i in formatted_param_names]
        auto_desc = f'{fun.__name__}: {param_names_pbar}'

        if len(auto_desc) > 40:
            auto_desc = auto_desc[:40] + '(...)]'

        desc = auto_desc

    # Execute the parallel run
    result: List[List[Any]]

    if backend == 'joblib':
        _log('Running under joblib', verbose)

        result, n_outputs = _parameterrun_joblib(fun, formatted_param_names, formatted_param_values,
                                                 list(indices_iterate), desc, n_workers=n_workers, pbar_bool=pbar_bool,
                                                 **kwargs)

    elif backend == 'mpi':
        _log(f'Running under mpi with {size} workers', verbose and rank == 0,
             hostname=mpi_module.Get_processor_name() if mpi_module is not None else None, )

        result, n_outputs = _parameterrun_mpi(fun, formatted_param_names, formatted_param_values, list(indices_iterate),
                                              desc, pbar_bool=pbar_bool, verbose=verbose, **kwargs)

    else:
        raise ValueError('Unknown backend')

    if rank == 0:
        final_result: Union[List[List[Any]], np.ndarray, List[np.ndarray]] = result

        # Reshape the result to get the (hyper)matrix
        if reshape:
            try:
                if n_outputs == 1:
                    final_result = _reshape_channel_result(result[0], n_values)
                else:
                    final_result = [_reshape_channel_result(result[i], n_values) for i in range(n_outputs)]
            except ValueError:
                print('Could not reshape the result')

        total_time = time() - time_start
        _log(f'Total time: {_normalized_time(total_time)}', verbose)

        if not result_as_array:
            final_result = _convert_result_to_lists(final_result)

        return final_result
    else:
        return None
