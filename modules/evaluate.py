#!/usr/bin/env python

# std libs
import os
import numpy as np
import numpy.linalg as la
import scipy.stats as stats
import pandas as pd

# own modules
import modules.aos as aos
import modules.gen as gen
import modules.data as data


log = gen.getLogger(__name__)

# -----------------------------------------------------------------------------
# evaluation helper functions
# -----------------------------------------------------------------------------
MAX_LAT_A_5p = np.array((4, -2, 1, 1, -1, 1, 0, 0, 1, 1, 1, 1, 4, 2, 1)).reshape(5, 3)
MAX_LAT_AI_5p = np.dot(la.inv(np.dot(np.transpose(MAX_LAT_A_5p),
                                     MAX_LAT_A_5p)), np.transpose(MAX_LAT_A_5p))


def max_lat(data):
    '''Return latitude with maximum wind speed for each model

    Parameters
    ----------
    data : np.array
        2-dimensional array (lat, model) with wind speeds.

    Returns
    -------
    np.array
        1-d array with latitude of maximum wind speed for each model.
    '''
    i0 = np.nanargmax(data, axis=0)
    i1 = np.arange(data.shape[1])

    d0 = data[i0, i1]
    dm = data[i0 - 1, i1]
    dmm = data[i0 - 2, i1]
    dp = data[i0 + 1, i1]
    dpp = data[i0 + 2, i1]

    d = np.vstack((dmm, dm, d0, dp, dpp))
    coeff = np.dot(MAX_LAT_AI_5p, d)
    x_max = i0 - 0.5 * coeff[1, :] / coeff[0, :]

    lat_max = x_max - 79.5
    return(lat_max)


def running_mean(x, N):
    '''Return the centered running mean over window-size N.

    Parameters
    ----------
    x : np.array
        2-dimensional array with data to calculate running avg over.
    N : int
        Size of averaging window.

    Returns
    -------
    np.array
        2-dimensional array with running averages.
    '''
    out = np.zeros_like(x)
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    out[N - 2:-(N - 2), :] = (cumsum[N:, :] - cumsum[:-N, :]) / float(N)
    return out


def _find_minima(data, rolling_window=3):
    '''Return indices to minima.

    Identify minima in the wind profile along the latitudes and return
    the indices to those minima. Smoothing via running_mean.

    **Note:** the running_mean creates a signal delay.

    Parameters
    ----------
    data : np.array
        2-dimensional (wind speed at latitude, model)
    rolling_window : int, optional
        Smoothing length, size of running mean window.

    Returns
    -------
    np.array
        2-dimensional array (minima, model) with flags to indicate minima.
    '''
    minima = np.zeros_like(data)

    # Duplicate data and fill nans to avoid comparison with nan
    data_fillna = data.copy()
    data_fillna = np.where(np.isnan(data_fillna), 0, data_fillna)

    # Calculate the running mean to smooth the curve
    data_fillna = running_mean(data_fillna, rolling_window)

    # Find the minima and return an array with zeros and ones where ones indicate minima.
    # Note that the signal is delayed due to the rolling average. But if the smoothing
    # window is reasonably sized compared to the length of the input array this shouldn't
    # matter much.
    minima[1:-1, :] = np.where(np.logical_and(data_fillna[0:-2] > data_fillna[1:-1],
                                              data_fillna[2:] > data_fillna[1:-1]), 1, 0)
    return minima


def max_lat_fumiaki(data):
    '''Find maximum based on the Fumiaki-approach.

    If the wind profile across the latitudes show a minimum above -66 degrees
    latitude then select the maximum below that as the maximum wind speed lat.
    Otherwise select the within range global maximum.

    Parameters
    ----------
    data : np.array
        2-dimensional array (wind speed at latitude, model)

    Returns
    -------
    np.array
        1-dimensional array with latitudes of max wind speed.
    '''
    minima = _find_minima(data, rolling_window=3)

    # Find indices for the minima if present and order those
    lat_ind, model_ind = np.nonzero(minima)
    order = np.argsort(model_ind)
    lat_ind = lat_ind[order].tolist()
    model_ind = model_ind[order].tolist()

    # If multiple minima exists select lower one. Here it would
    # be possible to add a check to ensure that if the minima
    # appears _too soon_ it can be ignored. That means, however,
    # that someone must define what "too soon" is.
    li = []
    mi = []
    for l, m in zip(lat_ind, model_ind):
        if m not in mi:
            mi.append(m)
            li.append(l)
        else:
            m_ind = mi.index(m)
            if l < li[m_ind]:
                li[m_ind] = l

    lat_ind = np.array(li)
    model_ind = np.array(mi)

    # lat_ind = 0 -> -79.5
    # So minima >= -66 means look for (first) minima at lat_ind > 13
    min_above_66th = model_ind[np.where(lat_ind > 13)]
    max_lats = np.ones(data.shape[1])

    # any models?
    if np.sum(min_above_66th) > 0.5:
        for m_ind in min_above_66th:
            cutoff = lat_ind[model_ind.tolist().index(m_ind)]
            if cutoff + 3 < data.shape[0]:
                cutoff += 3

            try:
                max_lats[m_ind] = max_lat(data[:cutoff, m_ind:m_ind + 1])[0]
            except IndexError:
                # Unexpected minima lead to index error.
                max_lats[m_ind] = np.inf

    inverse = [i for i in range(data.shape[1]) if i not in min_above_66th]
    max_lats[inverse] = max_lat(data[:, inverse])
    return max_lats


# -----------------------------------------------------------------------------
# Evaluation classes
# -----------------------------------------------------------------------------
class Evaluate:
    """Evaluate a single instance/window for all models.

    Calculate the correlation coefficient between the historic maximum
    latitude of the jet stream and the future predicted jet shift.

    Attributes
    ----------
    future_data_month_2d : np.array
    Future wind speeds
    historic_data_month_2d : np.array
    Historic wind speeds
    n_models : int
    Number of models
    """

    def __init__(self, historic_data_month_2d, future_data_month_2d, n_models):
        self.historic_data_month_2d = historic_data_month_2d
        self.future_data_month_2d = future_data_month_2d
        self.n_models = n_models
        # self.jet_shift = jet_shift

    def __call__(self, interval):
        """Function to execute to carry out the actual evaluation.

        Parameters
        ----------
        interval : np.array
            Array containing the start longitudes and
            averaging window size.

        Returns
        -------
        np.array
            Array with correlation coefficients and error flags.
        """
        # local imports needed when calling functions on different executors in parallel on Spark
        import modules.interval_x as intx
        import modules.evaluate as evaluate

        # Number of ensembles (window start position and averaging length combinations)
        n_ens = interval.shape[1]
        ret = np.zeros((3, n_ens))
        for k in range(n_ens):
            ind = interval[:, k:(k + 1)]

            # Start longitude and size of averaging window for current run
            start = ind[0, 0]
            length = ind[1, 0]

            # mask.shape: lon=360, lat=1, month=1, model=1
            lon_mask = intx.mask_360_start_length(ind[0:2, :]).reshape(360, 1, 1, 1)

            # month_mask.shape: lon=1, lat=1, month=24, model=1
            month_mask = intx.mask_months_24(ind, first='june', last='aug').reshape(-1)
            month_mask = month_mask.reshape(1, 1, 24, 1)

            # hist.shape: lon=360, lat=60, month=24, model=<n_model>
            hist = self.historic_data_month_2d.reshape(360, 60, 24, -1)
            fut = self.future_data_month_2d.reshape(360, 60, 24, -1)

            # mask_hist.shape: mask_hist.shape: lon=360, lat=60, month=24, model=<n_model>
            mask_hist = month_mask * lon_mask * hist
            mask_fut = month_mask * lon_mask * fut

            # find latitude with maximum windspeed (historic and future)
            # mask_hist_mean.shape: lat=60, model=<n_model>
            historic_lm = evaluate.max_lat_fumiaki(np.mean(mask_hist, axis=(0, 2)))
            future_lm = evaluate.max_lat_fumiaki(np.mean(mask_fut, axis=(0, 2)))

            # Calculate jet_shift. jet_shift.shape: model=<n_model>
            jet_shift = future_lm - historic_lm

            # lm.shape: model=<n_model>
            lm = historic_lm.copy()

            # Look for flags indicating errors
            if np.isinf(lm).any() or np.isinf(jet_shift).any():
                # Some models failed due to multiple minima/maxima issues
                models = []
                if np.isinf(lm).any():
                    models.extend([i for i, m in enumerate(lm.tolist()) if m == np.inf])
                if np.isinf(jet_shift).any():
                    models.extend([i for i, m in enumerate(jet_shift.tolist()) if m == np.inf])
                models = list(set(models))
                error_flag = sum([1 << m for m in models])
                ret[1, k] = error_flag

                # Set value of failed models to something. Initially 0, but average
                # of valid entries might be best for not shifting the mean jet position.
                # lm[np.where(lm == np.inf)] = 0
                lm[np.where(np.isinf(lm))] = lm[np.where(~np.isinf(lm))].mean()
                jet_shift[np.where(np.isinf(jet_shift))] = jet_shift[
                    np.where(~np.isinf(jet_shift))].mean()

            if np.isnan(lm).any() or np.isnan(jet_shift).any():
                # Some models failed due to multiple minima/maxima issues
                ret[0, k] = 2

                # flag errors
                models = []
                if np.isnan(lm).any():
                    models.extend([i for i, m in enumerate(lm.tolist()) if m == np.nan])
                if np.isnan(jet_shift).any():
                    models.extend([i for i, m in enumerate(jet_shift.tolist()) if m == np.nan])
                models = list(set(models))
                error_flag = sum([1 << m for m in models])
                ret[2, k] = error_flag
            else:
                # No (more) errors - calculate the correlation coefficient.
                ret[0, k] = stats.pearsonr(lm, jet_shift)[0]

        # return
        return (ret)


class EvaluateAll:
    """Class to evaluate all calculations.

    For each window, one evaluation is required, this class runs all the evaluations
    and processes the results which are all returned at the same time from a single
    call.

    Attributes
    ----------
    fitness : float
        The correlation value.
    length : np.array(int)
        Length of the longitude averaging window in degrees.
    max_length : int
        Maximum size of averaging window [1, 360).
    min_length : int
        Minumum size of averaging window [1, 359).
    minima_error_file : str
        Name of file to write minima-errors to.
    n : int
        Range of starting degrees [0, 360].
    nan_error_file : str
        Name of file to write nan-errors to.
    result : float
        Correlation coefficients from evaluation.
    ret : np.ndarray
        Array with 'raw' return values from evaluation.
    start : np.array(int)
        Start longitude for averaging window.
    """

    def __init__(self, e, n=360, max_length=360, min_length=30):
        """Class initialization

        Parameters
        ----------
        e : class
            Evaluation class for a single evaluation to run
        n : int, optional
            Range of starting degrees [0, 360].
        max_length : int, optional
            Maximum size of averaging window [1, 360).
        min_length : int, optional
            Minumum size of averaging window [1, 359).

        Returns
        -------
        TYPE
            Description
        """
        # logging
        log = gen.getLogger(gen.getClassName(self))
        log.info("start evaluating all [JJA]")

        # Error files
        self.minima_error_file = 'min_max_error.csv'
        self.nan_error_file = 'nan_error.csv'

        # aos
        obj = aos.DataObject(('EvaluateAll [JJA]', n))
        if obj.load():
            obj.copy(self)
            return(None)

        # prepare evaluate
        self.n = n
        self.min_length = min_length
        self.max_length = max_length

        x = np.arange(self.n)
        y = np.arange(self.min_length, self.max_length + 1)
        xx, yy = np.meshgrid(x, y)
        individuals = np.transpose(np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1))))

        # evaluate
        self.ret = e(individuals)

        # Process / convert results
        self.start = individuals[0, :].reshape(-1, 1)
        self.length = individuals[1, :].reshape(-1, 1)
        self.fitness = self.ret[0, :].reshape(-1, 1)
        self.result = pd.DataFrame({
            'start': self.start.ravel(),
            'length': self.length.ravel(),
            'fitness': self.fitness.ravel()},
            index=np.arange(len(self.fitness.ravel())))

        # Longitudes in input goes from -179.5 to 179.5.
        # Here we convert from integer degree (offset) to the proper longitude value.
        self.result['start'] -= 179.5
        self.save_errors(e.f.n_models)

        # aos
        obj.save(self)

    def _binary_flags_to_list(self, flags, n_models):
        """Convert a integer value into a list of zeros and ones.

        For n_models == 5, the binary number can be 00000 to 11111,
        where ones indicates errors. The flags are returned as an integer
        which is converted back into a list containing the number of the
        model(s) with errors.

        Parameters
        ----------
        flags : int
            Integer representation of the binary flags.
        n_models : int
            Number of models in system.

        Returns
        -------
        list(int)
            Collection of model numbers with errors.
        """
        items = []
        for row in range(flags.shape[0]):
            current = []

            # We loop in reverse (check left-most bits first)
            for i in range(n_models, -1, -1):
                # Check if bit `i` is == 1
                if ((1 << i) & int(flags[row])) > 0:
                    current.append(i)
            items.append(current)
        return items

    def _write_error_file(self, error_list, file, mode='w'):
        """Write errors to file.

        The errors are saved in a CSV-formatted file using semicolon, `;`,
        as the model delimiter, but `,` as column delimiter. The file contains
        four columns: start-longitude, length-of-window, ';'-separated list of
        integers (model numbers), and a ';'-separated list of model names.

        Parameters
        ----------
        error_list : list(int)
            List of model indices for models with errors.
        file : str
            Name of file to write errors to.
        mode : str, optional
            file access mode, 'w' for overwrite, 'a' for append.
        """
        with open(file, mode) as f:
            for i, models in enumerate(error_list):
                if len(models) > 0:
                    items = [
                        self.result.iloc[i].start,
                        self.result.iloc[i].length,
                        ';'.join([str(m) for m in models]),
                        ';'.join([data.NAMES_UA[m] for m in models])]
                    line = ','.join([str(item) for item in items])
                    f.write(line + os.linesep)

    def save_errors(self, n_models):
        """Save errors to files.

        There are two types of errors that are catched: min_max_errors and nan_errors.
        `min_max_errors` occur when minor local minima generate invalid maxima, thus
        making the computation fail. `nan_errors` occur if the maximum is too close to
        the boundary.

        The output filenames are specified as
        -------------------------------------
        self.minima_error_file = 'min_max_error.csv'
        self.nan_error_file = 'nan_error.csv'

        Parameters
        ----------
        n_models : int
            Number of models in ensemble.
        """
        min_max_errors = self._binary_flags_to_list(self.ret[1, :], n_models)
        nan_errors = self._binary_flags_to_list(self.ret[2, :], n_models)

        self._write_error_file(min_max_errors, self.minima_error_file)
        self._write_error_file(nan_errors, self.nan_error_file)
