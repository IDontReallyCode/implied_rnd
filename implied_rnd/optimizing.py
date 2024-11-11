from scipy.optimize import minimize
from scipy.stats import genpareto

import numpy as np


def evalgenpareto(theta: np.ndarray, x: np.ndarray, f: np.ndarray):
    # theta: Parameters for the Generalized Pareto distribution
    # x: Data points at which to evaluate the PDF
    # f: Observed frequencies or densities
    
    # Extract shape, location, and scale parameters from theta
    shape, loc, scale = theta
    # Calculate the PDF of the Generalized Pareto distribution at the given data points
    fhat = genpareto.pdf(x, c=shape, loc=loc, scale=scale)

    # Compute the sum of squared errors between the observed and estimated densities
    e2 = np.sum((fhat - f) ** 2)

    return e2


def evalgenbeta2(theta: np.ndarray, x: np.ndarray, f: np.ndarray):
    # print(theta)
    raise ValueError('Generalized Beta 2 is not ready')





def _fittail(x: np.ndarray, f: np.ndarray, evaldensity = evalgenpareto):
    initial_theta = [0.0, 0.0, 1.0]
    
    # Minimize the function
    result = minimize(evaldensity, initial_theta, method='Nelder-Mead', args=(x, f))
    initial_theta = result.x
    result = minimize(evaldensity, initial_theta, method='L-BFGS-B', args=(x, f))
    # result = minimize(_evalgenpareto, initial_theta, method='BFGS', args=(x, f))
    
    
    # The optimized theta values
    optimized_theta = result.x
    
    return optimized_theta