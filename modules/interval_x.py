#!/usr/bin/env python

# import
import numpy as np


# -----------------------------------------------------------------------------
# month definitions
# -----------------------------------------------------------------------------
months = {k: v for v, k in enumerate('jan feb mar apr may june july aug sept oct nov dec'.split())}


# -----------------------------------------------------------------------------
# Mask generation
# -----------------------------------------------------------------------------
def mask_360_start_length(x):
    """Create mask for complete 360 set with start position and window length.

    Parameters
    ----------
    x : np.array
        array with start and length values to create masks for.

    Returns
    -------
    np.array
        A masking array with zeros and ones.
    """
    y = np.zeros((360, x.shape[1]))
    for k in range(x.shape[1]):
        start = x[0, k]
        length = x[1, k]
        y[np.arange(start, start + length) % 360, k] = 1
    return(y)


def mask_months_24(x, first, last):
    """Mask the given months

    Parameters
    ----------
    x : np.array
        Data to base mask on
    first : str
        Abbrev. name of first month to mask (use).
    last : str
        Abbrev. name of last (inclusive) month to mask (use).

    Returns
    -------
    np.array
        Mask to apply.
    """
    if (first not in months.keys()) or (last not in months.keys()):
        msg = 'Months not a valid abbreviated months: {}, {}. '.format(first, last)
        msg += 'Valid months: {}'.format(', '.join([k for k in months.keys()]))
        raise ValueError(msg)

    start = months.get(first)
    end = months.get(last)

    if end < start:
        end += 12

    x = x * 11
    y = np.zeros((24, x.shape[1]))
    for k in range(x.shape[1]):
        y[start:(end + 1), k] = 1
    return(y)


def mask_360(x):
    x = x * 360
    y = np.zeros((360, x.shape[1]))
    for k in range(x.shape[1]):
        i0 = int(x[0, k])
        i1 = int(x[1, k])
        if i0 < i1:
            y[i0:i1, k] = 1
        else:
            y[:i1, k] = 1
            y[i0:, k] = 1
    return(y)


def mask_24(x):
    x = x * 24
    y = np.zeros((24, x.shape[1]))
    for k in range(x.shape[1]):
        i0 = int(x[2, k])
        i1 = int(x[3, k])
        if i1 < i0:
            (i0, i1) = (i1, i0)
        i0 = min(i0, 11)
        y[i0:i1, k] = 1
    return(y)


def mask_3_24(x):
    x = x * 11
    y = np.zeros((24, x.shape[1]))
    for k in range(x.shape[1]):
        i = int(x[2, k])
        i = min(i, 11)
        i = max(i, 0)
        y[i:(i + 3), k] = 1
    return(y)
