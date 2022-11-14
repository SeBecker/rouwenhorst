"""This module contains all functions for conducting the Rouwenhorst Method

The method is used for approximating the AR(1) income process of the model.
Contrary to other methody like the ones proposed by Tauchen 1986 as well as
Tauchen & Hussey 1991, this method is able to match the five important statistic
statistics (conditional/unconditional mean/variance as well as the first order 
autocorrelation).


For more information see Rouwenhorst 1995 & Kopecky & Suen 2010.
"""

from src.rouwenhorst import generate_grid, construct_trans_matrix

def generate_process(sigma, N, mu, rho):
    """This matrix generates a discrete grid as well as a transition matrix
    that characterizes the approximation of the AR(1) process 

    z_t = rho * z_t-1 + epsilon_t

    where epsilon_t is white noise with variance sigma_epsilon

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
    transition_matrix: np.array
        An NxN matrix whereas each row indicates the transition probabilities
        from the row specific state into the column specific state
    """


    return generate_grid(sigma, N, mu, rho), construct_trans_matrix(N, rho)