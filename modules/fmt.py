#!/usr/bin/env python

# std libs
import numpy as np


# -----------------------------------------------------------------------------
# functions
# -----------------------------------------------------------------------------
def reformat_1d_ens_2d_360_180(x):
    """Reformat a 1 dimensional ensemble into a 2 dimensional ensemble with a
    360 by 180 grid.

    Parameters
    ----------
    x : data.classes
        An instance of one of the data classes in the 'data' module.

    Returns
    -------
    tuple(list[])
        `data` and `area` as 2d numpy arrays

    Raises
    ------
    Exception
        Raises exception on invalid array dimensions/sizes.
    """
    if x.points.shape[1] != 2:
        raise Exception("reformat_2d_360_180: error")
    if x.area.shape[1] != 1:
        raise Exception("reformat_2d_360_180: error")
    if x.points.shape[0] != x.area.shape[0] or x.points.shape[0] != x.data.shape[0]:
        raise Exception("reformat_2d_360_180: error")

    n = x.points.shape[0]
    ij = np.array(x.points + (179.5, 89.5), dtype=int)
    if np.min(ij) != 0:
        raise Exception("reformat_2d_360_180: error")
    if np.max(ij[:, 0]) != 359:
        raise Exception("reformat_2d_360_180: error")
    if np.max(ij[:, 1]) != 179:
        raise Exception("reformat_2d_360_180: error")

    area = np.zeros((360, 180))
    if len(x.data.shape) == 2:
        # fully time averaged case
        data = np.zeros((360, 180, x.data.shape[1]))
        for k in range(n):
            area[ij[k, 0], ij[k, 1]] = x.area[k, 0]
            data[ij[k, 0], ij[k, 1], :] = x.data[k, :]
    elif len(x.data.shape) == 3:
        # month averaged case
        data = np.zeros((360, 180, x.data.shape[1], x.data.shape[2]))
        for k in range(n):
            area[ij[k, 0], ij[k, 1]] = x.area[k, 0]
            data[ij[k, 0], ij[k, 1], :, :] = x.data[k, :, :]
    else:
        raise Exception("reformat_2d_360_180: error")

    return([data, area])


def crop_2d_360_180_2d_360_60(l, lat_range=slice(10, 70)):
    """
    Parameters
    ----------
    l : list(np.array)
        list with data-array at position 0 and area-array at position 1.
    lat_range : slice, optional
        Range to crop latitudes to. First index is inclusive, second exclusive,
        as in [10, 12) = 10, 11.

    Returns
    -------
    tuple
        Tuple with list containing cropped data and area, respectively.

    Raises
    ------
    Exception
        If invalid number of dimensions of data-array.
    """
    data = l[0]
    area = l[1]
    if len(data.shape) == 3:
        # fully time averaged case
        return ([data[:, lat_range, :], area[:, lat_range]])
    elif len(data.shape) == 4:
        return ([data[:, lat_range, :, :], area[:, lat_range]])
    else:
        raise Exception("reformat_2d_360_180: error")


def avg_lon(data):
    '''Return np.nanmean along axis 0'''
    return(np.nanmean(data, axis=0))


def avg_lon_sl(data, start, length):
    """Return np.nanmean along axis 0, starting at

    Parameters
    ----------
    data : np.array
        Data to take average of
    start : int
        start longitude integer
    length : int
        size of averaging window in integers (degrees)

    Returns
    -------
    np.array
        averaged over window
    """
    end = (start + length) % 360
    if end < start:
        avg_window = np.concatenate((data[0:end], data[start:]), axis=0)
        return np.nanmean(avg_window, axis=0)
    return(np.nanmean(data[start:start + length], axis=0))


def avg_DJF(data):
    '''Average over the months dec, jan, feb'''
    assert(data.shape[1] == 24)
    return(np.nanmean(data[:, 11:14, :], axis=1))


def avg_JJA(data):
    '''Average over the months june, july, aug'''
    assert(data.shape[1] == 24)
    return(np.nanmean(data[:, 5:8, :], axis=1))


def avg_JD(data):
    assert(data.shape[1] == 24)
    return(np.nanmean(data[:, :12, :], axis=1))
