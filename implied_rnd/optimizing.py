from scipy.optimize import minimize
from scipy.stats import genpareto

import numpy as np


def evalgenpareto(theta: np.ndarray, x: np.ndarray, f: np.ndarray):
    # print(theta)
    # Create a Generalized Beta 2 distribution object
    # gbeta2_dist = genpareto(a=theta[0], b=theta[1], p=theta[2])
    fhat = genpareto.pdf(x,theta[0], loc=theta[1])
    e2 = np.sum((fhat-f)**2)
    # print(e2)
    return e2


def evalgenbeta2(theta: np.ndarray, x: np.ndarray, f: np.ndarray):
    # print(theta)
    raise ValueError('Generalized Beta 2 is not ready')





def _fittail(x: np.ndarray, f: np.ndarray, evaldensity = evalgenpareto):
    initial_theta = [1.0, 0.0]
    
    # Minimize the function
    result = minimize(evaldensity, initial_theta, method='Nelder-Mead', args=(x, f))
    initial_theta = result.x
    result = minimize(evaldensity, initial_theta, method='L-BFGS-B', args=(x, f))
    # result = minimize(_evalgenpareto, initial_theta, method='BFGS', args=(x, f))
    
    
    # The optimized theta values
    optimized_theta = result.x
    
    return optimized_theta