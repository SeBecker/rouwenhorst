"""This module contains the functions that set up the transition matrix as well
as the associated grid for approximating a highly persistent AR(1) process via 
the Rouwenhorst method.
"""

import numpy as np
from numba import njit


@njit()
def construct_trans_matrix(N, rho):
    """This function derives the matrix of transition probabilities for the
    Markov chain

    Parameters
    ----------
    N: float
       The number of grid points that should be used to approximate the process
    rho: float
        parameter that describes the autocorrelation of the process

    Returns
    ------
    trans_mat: np.array
        An NxN matrix whereas each row indicates the transition probabilities
        out of the specific state
    """
    # variance of the AR process

    p = (1 + rho) / 2
    # gen transition matrix

    trans_mat = np.array([[p, 1 - p], [1 - p, p]])

    for n in range(3, N + 1):
        part1, part2, part3, part4 = (
            generate_mat_ext(trans_mat, n, 1),
            generate_mat_ext(trans_mat, n, 2),
            generate_mat_ext(trans_mat, n, 3),
            generate_mat_ext(trans_mat, n, 4),
        )

        trans_mat = p * part1 + (1 - p) * part2 + (1 - p) * part3 + p * part4
        trans_mat[1 : n - 1, :] = trans_mat[1 : n - 1, :] / 2
    return trans_mat


@njit()
def generate_mat_ext(mat1, step_n, form):
    """This function creates the matrices according to step 2 of the process
    described by Kopecky & Suen 2010.

    Parameters
    ----------
    mat1: np.array
       Trans matrix from previous step
    step_N: int
       current number of iteration
    form: int
       indicator which of the extended matrices should be created

    Returns
    ------
    processed_trans_mat: np.array
        A step_N x step_N matrix that will be used to compute the transition matric
    """

    if form == 1:
        return np.append(
            np.append(mat1, np.zeros((step_n - 1, 1)), axis=1),
            np.zeros((1, step_n)),
            axis=0,
        )
    elif form == 2:
        return np.append(
            np.append(np.zeros((step_n - 1, 1)), mat1, axis=1),
            np.zeros((1, step_n)),
            axis=0,
        )
    elif form == 3:
        return np.append(
            np.zeros((1, step_n)),
            np.append(mat1, np.zeros((step_n - 1, 1)), axis=1),
            axis=0,
        )
    elif form == 4:
        return np.append(
            np.zeros((1, step_n)),
            np.append(np.zeros((step_n - 1, 1)), mat1, axis=1),
            axis=0,
        )


@njit()
def generate_grid(sigma, N, mu, rho):
    """This function derives the matrix of transition probabilities for the
    Markov chain

    Parameters
    ----------
    sigma: float
        standard deviation of the epsilon value
    N: float
       The number of grid points that should be used for the approximation
    mu: float
        unconditional mean of the process
    rho: float
        parameter that describes the autocorrelation of the process

    Returns
    ------
    grid: np.array
        A 1xN array that contains the grid values
    """

    sigma_p = sigma / np.sqrt((1 + rho ** 2))

    x = np.sqrt(N - 1) * sigma_p
    return np.linspace(-x, x, N) + mu

