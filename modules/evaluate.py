#!/usr/bin/env python

# std libs
import numpy as np
import numpy.linalg as la

# own modules
import modules.gen as gen


log = gen.getLogger(__name__)

# -----------------------------------------------------------------------------
# evaluation
# -----------------------------------------------------------------------------
MAX_LAT_A_5p = np.array((4, -2, 1, 1, -1, 1, 0, 0, 1, 1, 1, 1, 4, 2, 1)).reshape(5, 3)
MAX_LAT_AI_5p = np.dot(la.inv(np.dot(np.transpose(MAX_LAT_A_5p),
                                     MAX_LAT_A_5p)), np.transpose(MAX_LAT_A_5p))


def max_lat(data):
    # data.shape: lat, model
    # print('max_lat data shape: ', data.shape)
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

    # return latitude maximum positon for each model
    return(lat_max)


def running_mean(x, N):
    '''Return the centered running mean over window-size N'''
    out = np.zeros_like(x)
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0)
    out[N - 2:-(N - 2), :] = (cumsum[N:, :] - cumsum[:-N, :]) / float(N)
    return out


def _find_minima(data, rolling_window=3):
    '''Return indices to minima'''
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
    '''Find maximum based on the Fumiaki-approach'''
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
                # Unexpected minimas lead to index error.
                max_lats[m_ind] = np.inf

    inverse = [i for i in range(data.shape[1]) if i not in min_above_66th]
    max_lats[inverse] = max_lat(data[:, inverse])
    return max_lats
