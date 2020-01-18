import numpy as np
import pyemd
from scipy.linalg import toeplitz
from sklearn.metrics.pairwise import pairwise_kernels
from eden.graph import vectorize


def emd(x, y, distance_scaling=1.0):
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float)
    distance_mat = d_mat / distance_scaling

    emd_val = pyemd.emd(x, y, distance_mat)
    return emd_val


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """
    Gaussian kernel with squared distance in exponential term replaced by EMD
    Args:
    x, y: 1D pmf of two distributions with the same support
    sigma: standard deviation
    """

    # Calculate emd
    emd_val = emd(x, y, distance_scaling=distance_scaling)

    return np.exp(-1 * emd_val * emd_val / (2 * sigma * sigma))


def gaussian(x, y, sigma=1.0):
    dist = np.linalg.norm(x - y, 2)
    return np.exp(-1 * dist * dist / (2 * sigma * sigma))


def kernel_compute(X, Y=None, is_hist=True, metric='linear', n_jobs=None):

    def preprocess(X, max_len, is_hist):
        X_p = np.zeros((len(X), max_len))
        for i in range(len(X)):
            X_p[i, :len(X[i])] = X[i]

        if is_hist:
            row_sum = np.sum(X_p, axis=1)
            X_p = X_p / row_sum[:, None]

        return X_p

    if metric == 'nspdk':
        X = vectorize(X, complexity=4, discrete=True)

        if Y is not None:
            Y = vectorize(Y, complexity=4, discrete=True)

        return pairwise_kernels(X, Y, metric='linear', n_jobs=n_jobs)

    else:
        max_len = max([len(x) for x in X])
        if Y is not None:
            max_len = max(max_len, max([len(y) for y in Y]))

        X = preprocess(X, max_len, is_hist)

        if Y is not None:
            Y = preprocess(Y, max_len, is_hist)

        return pairwise_kernels(X, Y, metric=metric, n_jobs=n_jobs)


def compute_mmd(samples1, samples2, metric, is_hist=True, n_jobs=None):
    """
    MMD between two list of samples
    """

    # print('Compute_mmd', inspect.ArgSpec(compute_mmd))

    X = kernel_compute(samples1, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Y = kernel_compute(samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Z = kernel_compute(samples1, Y=samples2, is_hist=is_hist,
                       metric=metric, n_jobs=n_jobs)

    return np.average(X) + np.average(Y) - 2 * np.average(Z)
