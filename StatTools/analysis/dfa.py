import time
from contextlib import closing
from ctypes import c_double
from functools import partial
from math import exp, floor
from multiprocessing import Pool, cpu_count, Array, Lock, Value
from threading import Thread
from typing import Union, Tuple

import numpy as np
from tqdm import TqdmWarning, tqdm


# ====================== DFA core worker ======================

def _detrend_segment(y_segment: np.ndarray, indices: np.ndarray, degree: int) -> np.ndarray:
    """
    Compute detrended residuals for a single segment.
    
    Args:
        y_segment: Integrated time series segment.
        indices: Indices for polynomial fitting.
        degree: Polynomial degree for detrending.
    
    Returns:
        Residuals (detrended segment).
    """
    coef = np.polyfit(indices, y_segment, deg=degree)
    trend = np.polyval(coef, indices)
    residuals = y_segment - trend
    return residuals


def _fluctuation_function(f_q_s_sum: float, cycles_amount: int, degree: int) -> float:
    """
    Compute fluctuation function F(s) from accumulated sum.
    
    Args:
        f_q_s_sum: Accumulated sum of (f2)^(degree/2) over all segments.
        cycles_amount: Number of segments processed.
        degree: Polynomial degree used in detrending.
    
    Returns:
        Fluctuation function F(s).
    """
    return ((1.0 / cycles_amount) * f_q_s_sum) ** (1.0 / degree)


def dfa_worker(
        indices: Union[int, list, np.ndarray],
        arr: Union[np.ndarray, None] = None,
        degree: int = 2,
        s_values: Union[list, np.ndarray, None] = None,
) -> list:
    """
    Core of the DFA algorithm. Processes a subset of series (indices) and
    returns fluctuation functions F(s).

    Args:
        indices: Indices of time series in the dataset to process.
        arr: Dataset array (must be 2D, shape: (n_series, length)).
        degree: Polynomial degree for detrending.
        s_values: Pre-calculated box sizes (scales).

    Returns:
        list of (s, F_s) for each requested index.
    """
    data = np.asarray(arr, dtype=float)

    if data.ndim != 2:
        raise ValueError(
            f"dfa_worker expects 2D array, got {data.ndim}D array. "
            f"Normalize data to 2D before calling (use reshape(1, -1) for 1D)."
        )

    if not isinstance(indices, (list, np.ndarray)):
        indices = [indices]

    series_len_global = data.shape[1]
    if s_values is None:
        s_max = int(series_len_global / 4)
        s_values = [int(exp(step)) for step in np.arange(1.6, np.log(s_max), 0.5)]
    else:
        s_values = list(s_values)

    results = []

    for idx in indices:
        series = data[idx]

        # Standard DFA preprocessing: mean-centering and integration
        data_centered = series - np.mean(series)
        y_cumsum = np.cumsum(data_centered)
        series_len = len(data_centered)

        s_list = []
        F_list = []

        for s_val in s_values:
            if s_val >= series_len / 4:
                continue

            s = np.linspace(1, s_val, s_val, dtype=int)
            len_s = len(s)
            cycles_amount = floor(series_len / len_s)
            if cycles_amount < 1:
                continue

            f_q_s_sum = 0.0
            s_temp = s.copy()

            for i in range(1, cycles_amount):
                indices_s = np.array((s_temp - (i + 0.5) * len_s), dtype=int)
                y_cumsum_s = np.take(y_cumsum, s_temp)

                # Compute detrended residuals for this segment
                residuals = _detrend_segment(y_cumsum_s, indices_s, degree)

                # Mean squared residual 
                f2 = np.sum(residuals ** 2) / len_s

                # Accumulate according to the definition of F_q(s)
                f_q_s_sum += f2 ** (degree / 2)
                s_temp += s_val

            # Compute fluctuation function F(s)
            F_s = _fluctuation_function(f_q_s_sum, cycles_amount, degree)
            s_list.append(s_val)
            F_list.append(F_s)

        results.append((np.array(s_list), np.array(F_list)))

    return results


# ====================== High-level DFA function ======================

def dfa(
        dataset,
        degree: int = 2,
        processes: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the Detrended Fluctuation Analysis (DFA) method.

    The algorithm removes local polynomial trends in integrated time series and
    returns the fluctuation function F(s) for each series.

    Args:
        dataset (ndarray): 1D or 2D array of time series data.
        degree (int): Polynomial degree for detrending (default: 2).
        processes (int): Number of parallel workers (default: 1).

    Returns:
        tuple: (s, F_s)
            - For 1D input: two 1D arrays s, F_s.
            - For 2D input:
                s is a 1D array (same scales for all series),
                F_s is a 2D array where each row is F(s) for one time series.
    """
    data = np.asarray(dataset, dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)
        single_series = True
    elif data.ndim == 2:
        single_series = False
    else:
        raise ValueError("Only 1D or 2D arrays are allowed!")

    series_len = data.shape[1]
    s_max = int(series_len / 4)
    s_values = [int(exp(step)) for step in np.arange(1.6, np.log(s_max), 0.5)]

    n_series = data.shape[0]
    results = None

    if processes <= 1:
        indices = np.arange(n_series)
        results = dfa_worker(
            indices=indices,
            arr=data,
            degree=degree,
            s_values=s_values,
        )
    else:
        processes = min(processes, cpu_count(), n_series)
        chunks = np.array_split(np.arange(n_series), processes)

        worker_func = partial(
            dfa_worker,
            arr=data,
            degree=degree,
            s_values=s_values,
        )

        results_list_of_lists = []
        with closing(Pool(processes=processes)) as pool:
            for sub in pool.map(worker_func, chunks):
                results_list_of_lists.append(sub)

        flat_results = []
        for sub in results_list_of_lists:
            flat_results.extend(sub)
        results = flat_results

    s_list = [r[0] for r in results]
    F_list = [r[1] for r in results]

    if single_series:
        s_out = s_list[0]
        F_out = F_list[0]
    else:
        s_out = s_list[0]
        F_out = np.vstack(F_list)

    return s_out, F_out


# ====================== Progress bar manager ======================

def bar_manager(description, total, counter, lock, mode="total", stop_bit=None):
    """
    Manages progress bar display for long-running operations.

    Args:
        description (str): Description text for the progress bar.
        total (int): Total number of items to process.
        counter (Value): Shared counter for tracking progress.
        lock (Lock): Thread lock for safe counter access.
        mode (str): Display mode - "total" (absolute) or "percent".
        stop_bit (Value): Optional stop signal for early termination.

    Returns:
        None
    """
    max_val = total
    if mode == "percent":
        max_val = 100
    with closing(tqdm(desc=description, total=max_val, leave=False, position=0)) as bar:
        try:
            last_val = counter.value
            while True:
                if stop_bit is not None and stop_bit.value > 0:
                    break

                time.sleep(0.25)
                with lock:
                    if counter.value > last_val:
                        if mode == "percent":
                            bar.update(
                                round(((counter.value - last_val) * 100 / total), 2)
                            )
                        else:
                            bar.update(counter.value - last_val)
                        last_val = counter.value

                    if counter.value == total:
                        bar.close()
                        break
        except TqdmWarning:
            return None


# ====================== DFA wrapper class ======================

class DFA:
    """
    Detrended Fluctuation Analysis (DFA) wrapper class for estimating the Hurst exponent.
    """

    def __init__(self, dataset, degree=2, root=False, ignore_input_control=False):
        """
        Initialize DFA analyzer.

        Args:
            dataset: Input time series data (1D or 2D array, or file path).
            degree: Polynomial degree for detrending (default: 2).
            root: Kept for backward compatibility, not used in current implementation.
            ignore_input_control: Skip input validation (default: False).
        """
        self.degree = degree
        self.root = root  
        self.dataset = None
        self.s = None
        self.F_s = None

        if ignore_input_control:
            s_vals, F_vals = dfa(
                dataset,
                degree=self.degree,
                processes=1,
            )
            self.s = s_vals
            self.F_s = F_vals
        else:
            if isinstance(dataset, str):
                try:
                    dataset = np.loadtxt(dataset)
                except OSError:
                    raise NameError("\n    The file either doesn't exist or the path is wrong!")
                if np.size(dataset) == 0:
                    raise NameError("\n    Input file is empty!")

            if not isinstance(dataset, np.ndarray):
                try:
                    dataset = np.array(dataset, dtype=float)
                except (ValueError, TypeError):
                    raise NameError("\n    Input dataset is supposed to be numpy array, list or filepath!")

            if dataset.ndim > 2 or dataset.ndim == 0:
                raise NameError("\n    Only 1- or 2-dimensional arrays are allowed!")

            series_len = len(dataset) if dataset.ndim == 1 else dataset.shape[1]
            if series_len < 20:
                raise NameError("Wrong input array ! (It's probably too short)")

            self.dataset = np.array(dataset)

    @staticmethod
    def _hurst_exponent(x_axis: np.ndarray, y_axis: np.ndarray, simple_mode: bool = True) -> float:
        """
        Calculate the Hurst exponent (scaling exponent) by fitting
        y_axis vs x_axis with a linear regression.
        Here we assume x_axis = log(s), y_axis = log(F(s)).
        """
        if simple_mode:
            return np.polyfit(x_axis, y_axis, deg=1)[0]
        else:
            error_str = "\n    Non-linear approximation is not supported yet!"
            raise NameError(error_str)

    @staticmethod
    def initializer_for_parallel_mod(shared_array, h_est, shared_c, shared_l):
        """
        Initialize global variables for parallel processing.
        """
        global datasets_array
        global estimations
        global shared_counter
        global shared_lock
        datasets_array = shared_array
        estimations = h_est
        shared_counter = shared_c
        shared_lock = shared_l

    @staticmethod
    def dfa_core_cycle(dataset, degree, root):
        """
        Perform DFA for a single vector and return (s, F_s).

        Args:
            dataset: Input time series data (1D or 2D).
            degree: Polynomial degree for detrending.
            root: Kept for backward compatibility, not used.
        """
        data = np.asarray(dataset, dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        result_list = dfa_worker(
            indices=0,
            arr=data,
            degree=degree,
            s_values=None,
        )
        s_vals, F_vals = result_list[0]
        return s_vals, F_vals

    def find_h(self, simple_mode=True):
        """
        Estimate the Hurst exponent from fluctuation analysis.

        External interface is preserved:
        - returns scalar for 1D input;
        - returns 1D array for 2D input.

        Args:
            simple_mode: Kept for backward compatibility, not used
                         (always uses linear fit).
        """
        if not simple_mode:
            error_str = "\n    Non-linear approximation is not supported yet!"
            raise NameError(error_str)

        if self.dataset is None:
            raise NameError("\n    Dataset is not initialized for DFA.find_h!")

        s_vals, F_vals = dfa(
            self.dataset,
            self.degree,
            processes=1,
        )
        self.s, self.F_s = s_vals, F_vals

        if self.dataset.ndim == 1:
            log_s = np.log(self.s)
            log_F = np.log(self.F_s)
            h_value = self._hurst_exponent(log_s, log_F, simple_mode=True)
            return h_value
        else:
            n_series = self.dataset.shape[0]
            log_s = np.log(self.s)  
            h_array = np.empty(n_series, dtype=float)
            for i in range(n_series):
                log_F = np.log(self.F_s[i])
                h_array[i] = self._hurst_exponent(log_s, log_F, simple_mode=True)
            return h_array

    def parallel_2d(
            self,
            threads=cpu_count(),
            progress_bar=False,
            h_control=False,
            h_target=float(),
            h_limit=float(),
    ):
        """
        Parallel computation of Hurst exponents for 2D datasets.
        """
        if threads == 1 or self.dataset.ndim == 1:
            return self.find_h()

        if len(self.dataset) / threads < 1:
            print(
                "\n    DFA Warning: Input array is too small for using it in parallel mode!"
                f"\n    You should either use fewer threads ({len(self.dataset)}) or avoid parallel mode!"
            )
            return self.find_h()

        vectors_indices_by_threads = np.array_split(
            np.linspace(0, len(self.dataset) - 1, len(self.dataset), dtype=int),
            threads,
        )

        dataset_to_memory = Array(c_double, len(self.dataset) * len(self.dataset[0]))
        h_estimation_in_memory = Array(c_double, len(self.dataset))

        np.copyto(
            np.frombuffer(dataset_to_memory.get_obj()).reshape(
                (len(self.dataset), len(self.dataset[0]))
            ),
            self.dataset,
        )

        shared_counter = Value("i", 0)
        shared_lock = Lock()

        if progress_bar:
            bar_thread = Thread(
                target=bar_manager,
                args=("DFA", len(self.dataset), shared_counter, shared_lock),
            )
            bar_thread.start()

        with closing(
                Pool(
                    processes=threads,
                    initializer=self.initializer_for_parallel_mod,
                    initargs=(
                        dataset_to_memory,
                        h_estimation_in_memory,
                        shared_counter,
                        shared_lock,
                    ),
                )
        ) as pool:
            invalid_i = pool.map(
                partial(
                    self.parallel_core,
                    quantity=len(self.dataset),
                    length=len(self.dataset[0]),
                    h_control=h_control,
                    h_target=h_target,
                    h_limit=h_limit,
                ),
                vectors_indices_by_threads,
            )

        if progress_bar:
            bar_thread.join()

        if h_control:
            invalid_i = np.concatenate(invalid_i)
            return np.frombuffer(h_estimation_in_memory.get_obj()), invalid_i
        else:
            return np.frombuffer(h_estimation_in_memory.get_obj())

    def parallel_core(self, indices, quantity, length, h_control, h_target, h_limit):
        """
        Core function for parallel computation of Hurst exponents.
        """
        invalid_i = []

        all_data = np.frombuffer(datasets_array.get_obj()).reshape((quantity, length))

        for i in indices:
            vector = all_data[i]

            # Normalize to 2D format (single row)
            vector_2d = vector.reshape(1, -1)

            result_list = dfa_worker(
                indices=0,
                arr=vector_2d,
                degree=self.degree,
                s_values=None,
            )
            s_vals, F_vals = result_list[0]
            log_s = np.log(s_vals)
            log_F = np.log(F_vals)
            h_calc = self._hurst_exponent(log_s, log_F, simple_mode=True)

            np.frombuffer(estimations.get_obj())[i] = h_calc

            with shared_lock:
                shared_counter.value += 1

            if h_control and abs(h_calc - h_target) > h_limit:
                invalid_i.append(i)

        return np.array(invalid_i)