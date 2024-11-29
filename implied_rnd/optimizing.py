from scipy.optimize import minimize
from scipy.stats import genpareto
from scipy.stats import genextreme
from scipy.integrate import simps
from scipy.optimize import differential_evolution


import matplotlib.pyplot as plt

import numpy as np

F_GENPARETO = 0
F_GENEXTREME = 1


def evalgenpareto(theta: np.ndarray, x: np.ndarray, f: np.ndarray):
    # theta: Parameters for the Generalized Pareto distribution
    # x: Data points at which to evaluate the PDF
    # f: Observed frequencies or densities

    if len(theta) == 2:
        # Extract shape, location, and scale parameters from theta
        shape, scale = theta
        loc = 0
    else:
        shape, loc, scale = theta
    # Calculate the PDF of the Generalized Pareto distribution at the given data points
    fhat = genpareto.pdf(x, c=shape, loc=loc, scale=scale)

    # Compute the sum of squared errors between the observed and estimated densities
    e2 = np.sum((fhat - f) ** 2)

    return e2


def evalgenextreme(theta: np.ndarray, x: np.ndarray, f: np.ndarray):
    # theta: Parameters for the Generalized Extreme Value distribution
    # x: Data points at which to evaluate the PDF
    # f: Observed frequencies or densities

    if len(theta) == 2:
        # Extract shape, location, and scale parameters from theta
        shape, scale = theta
        loc = 0
    else:
        shape, loc, scale = theta

    # Create a GEV distribution object
    gevdist = genextreme(shape, loc, scale)
    # Calculate the PDF of the Generalized Extreme Value distribution at the given data points
    fhat = gevdist.pdf(x, c=shape, loc=loc, scale=scale)

    # Compute the sum of squared errors between the observed and estimated densities
    e2 = np.sum((fhat - f) ** 2)

    return e2


def evalgenbeta2(theta: np.ndarray, x: np.ndarray, f: np.ndarray):
    # print(theta)
    raise ValueError('Generalized Beta 2 is not ready')


def _fittailsandintegral(theta: np.ndarray, outputx: np.ndarray, interpmask: np.ndarray, extlftmask: np.ndarray, extrgtmask: np.ndarray, outputf: np.ndarray, whichdensity: int = F_GENPARETO):

    if whichdensity == F_GENPARETO:
        thisdensity = evalgenpareto
    elif whichdensity == F_GENEXTREME:
        thisdensity = evalgenextreme

    # The data that does not change, determined by Breenden and Litzenberger (1978)
    xinterp = outputx[interpmask]
    yinterp = outputf[interpmask]

    # Calculate the area under the curve
    # midlpartarea = simps(yinterp, xinterp)

    # 
    # Left tail **************************************************
    # the x data to be plotted
    xlefttail = outputx[extlftmask]
    # the x data to be fitted
    xlefttailfit = outputx[interpmask][0:2][::-1]
    refpoint = outputx[interpmask][1]
    xlefttailfit = -1*(xlefttailfit - refpoint)
    ylefttailfit = outputf[interpmask][0:2][::-1]

    xlefttaileval = -1*(xlefttail - refpoint)

    # Let us fit the scale and shape parameters of the Generalized Pareto Distribution, leaving location = 0
    # thetaleftnaive = _fittail(xlefttailfit, ylefttailfit)

    # Right tail **************************************************
    xrighttail = outputx[extrgtmask]

    xrighttailfit = outputx[interpmask][-2:]
    refpoint = outputx[interpmask][-2]
    xrighttailfit = xrighttailfit - refpoint
    yrighttailfit = outputf[interpmask][-2:]
    xrighttaileval = xrighttail - refpoint

    # thetarigtnaive = _fittail(xrighttailfit, yrighttailfit)

    # theta_left is the first three points of theta
    theta_lft = theta[0:3]
    # theta_right is the last three points of theta
    theta_rgt = theta[3:]

    # Evaluate the Generalized Pareto density for the left tail
    e2_lft = thisdensity(theta_lft, xlefttailfit, ylefttailfit)
    # Evaluate the Generalized Pareto density for the right tail
    e2_rgt = thisdensity(theta_rgt, xrighttailfit, yrighttailfit)

    outputf[extlftmask] = (genpareto.pdf(xlefttaileval, c=theta_lft[0], loc=theta_lft[1], scale=theta_lft[2]))
    outputf[extrgtmask] = (genpareto.pdf(xrighttaileval, c=theta_rgt[0], loc=theta_rgt[1], scale=theta_rgt[2]))

    # Now, I have outputx and outputf. I need to integrate the density over the entire range of outputx to make sure the integral is 1
    # I will use the trapezoidal rule to integrate the density
    # Compute the integral of the density
    integral = simps(outputx, outputf)
    # print(f'Integral: {integral}')

    # Compute the sum of squared errors between the integral and 1
    e2_integral = (integral - 1) ** 2

    # plt.plot(outputx, outputf)
    # plt.show()

    # Compute the sum of squared errors for the interpolation region and the tails
    e2 = e2_integral + e2_lft + e2_rgt

    # Return the sum of squared errors
    return e2


def fittails(outputx: np.ndarray, interpmask: np.ndarray, extlftmask: np.ndarray, extrgtmask: np.ndarray, outputf: np.ndarray, whichdensity: int = F_GENPARETO):

    # We fit both tails using two points each, while making sure the integral of the density is 1
    initial_theta = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    # Left tail **************************************************
    # the x data to be plotted
    xlefttail = outputx[extlftmask]
    # the x data to be fitted
    xlefttailfit = outputx[interpmask][0:2][::-1]
    refpoint = outputx[interpmask][1]
    xlefttailfit = -1*(xlefttailfit - refpoint)
    ylefttailfit = outputf[interpmask][0:2][::-1]

    xlefttaileval = -1*(xlefttail - refpoint)


    # Let us fit the scale and shape parameters of the Generalized Pareto Distribution, leaving location = 0
    thetaleft = _fittail(xlefttailfit, ylefttailfit)


    # Right tail **************************************************
    xrighttail = outputx[extrgtmask]

    xrighttailfit = outputx[interpmask][-2:]
    refpoint = outputx[interpmask][-2]
    xrighttailfit = xrighttailfit - refpoint
    yrighttailfit = outputf[interpmask][-2:]
    xrighttaileval = outputx[extrgtmask] - refpoint

    thetarigt = _fittail(xrighttailfit, yrighttailfit)

    initial_theta[0] = thetaleft[0]
    initial_theta[2] = thetaleft[1]
    initial_theta[3] = thetarigt[0]
    initial_theta[5] = thetarigt[1]

    # # Minimize the function
    # bounds__theta = [(0.0, None), (None, 0), (0.0, None), (0.0, None), (None, 0), (0.0, None)]
    # result = minimize(_fittailsandintegral, initial_theta, method='L-BFGS-B', 
    #                   bounds=bounds__theta, args=(outputx, interpmask, extlftmask, extrgtmask, outputf, whichdensity), 
    #                   options={'ftol': 1e-10, 'maxiter': 10000})

    # theta = [c1, loc1, scale1, c2, loc2, scale2]
    # Minimize the function using differential_evolution
    bounds__theta = [(0.0, 99999999), (-500, 0), (0.0, 99999999), (0.0, 99999999), (-500, 0), (0.0, 99999999)]
    result = differential_evolution(_fittailsandintegral, bounds__theta, 
                                    args=(outputx, interpmask, extlftmask, extrgtmask, outputf, whichdensity),
                                    tol=1e-10, maxiter=10000)
    

    # The optimized theta values check the output from the fitted fnction
    optimum = _fittailsandintegral(result.x, outputx, interpmask, extlftmask, extrgtmask, outputf, whichdensity)
    print(f'Optimum: {optimum}')
    # show the results theta
    print(f'Theta: {result.x}')


    return outputf


def _fittail(x: np.ndarray, f: np.ndarray, evaldensity = evalgenpareto):
    # check how many points in x
    # if there are two points, then we fit the data on two points and set the initial theta to 1.0, 1.0
    # if there are three points or more, then we fit on three points and set the initial theta to 1.0, 0.0, 1.0
    if len(x) == 2:
        initial_theta = [1.0, 1.0]
    elif len(x) > 2:
        initial_theta = [1.0, 0.0, 1.0]
    
    # Minimize the function
    result = minimize(evaldensity, initial_theta, method='Nelder-Mead', args=(x, f))
    initial_theta = result.x
    result = minimize(evaldensity, initial_theta, method='L-BFGS-B', args=(x, f))
    # result = minimize(_evalgenpareto, initial_theta, method='BFGS', args=(x, f))
    
    
    # The optimized theta values
    optimized_theta = result.x
    
    return optimized_theta