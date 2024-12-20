"""
    Based on Breeden and Litzenberger (1986), we can infer the risk-neutral density from the prices of options.
    The theory applies to European options.
    For non-dividend paying equity, using call prices, we can assume no early exercise and assume we can use Breeden and Litzenberger




    You need to provide two arrays: strikes, and IVs from OTM/ATM calls and puts
    
    There will be a default method applied, but you can select the method you want.
    
    3 general approaches: 1- standard support + extrapolation of IV, 2- standard support + extrapolation of density, 3- tslm support + extrapolation of IV
"""

import numpy as np
from numba import njit
from typing import Union, List
from py_vollib_vectorized import vectorized_black_scholes as bls
# from optimizing import _fitgenpareto
import implied_rnd.optimizing as opt
from scipy.stats import genpareto
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.integrate import simpson

import matplotlib.pyplot as plt
# import warnings

METHOD_STDR_EXTRAPIV = 0    # Standard support with extrapolation of IV
METHOD_STDR_EXTRADEN = 1    # Standard support with extrapolation of density
METHOD_TLSM_EXTRAPIV = 2    # Time-Scaled-Log-Moneyness support with extrapolation of IV using asymptotes

"""
    METHOD_STDR_EXTRAPIV
    We interpolate with some polynomial order, and extrapolate with a constant.
    
    METHOD_STDR_EXTRADEN
    We interpolate with some method, and extrapolate the density, for example, with a Generalized Pareto.
    
    METHOD_TLSM_EXTRAPIV
    We interpolate with some method, and extrapolate the IV with the asymptotes of the model.
"""

INTERP_LINEAR = 0
INTERP_POLYM1 = 11
INTERP_POLYM2 = 12
INTERP_POLYM3 = 13
INTERP_POLYM4 = 14
INTERP_POLYM5 = 15
INTERP_POLYM6 = 16
INTERP_POLYM7 = 17
INTERP_POLYM8 = 18
INTERP_POLYM9 = 19

INTERP_FACTR1 = 121      # Just one hyperbolla
INTERP_FACTR2 = 122      # Two different hyperbollas
INTERP_FACTR3 = 123      # Two hyperbollas + x for an asymetric feature
INTERP_FACTR31 = 231    # Two hyperbollas (centered at min value) + x for an asymetric feature
INTERP_FACTR4 = 124      # Two hyperbollas + arctan(x) for an asymetric/distortion feature
INTERP_FACTR5 = 125      # Two hyperbollas + x for asymetry + arctan(x) for an asymetric/distortion feature
INTERP_FACTR51 = 251    # four hyperbollas + x for asymetry + arctan(x) for an asymetric/distortion feature
INTERP_FACTR52 = 252    # four hyperbollas + x for asymetry + three arctan(x) for an asymetric/distortion feature

INTERP_FACTR6 = 126      # Just one hyperbolla + x for asymetry 
INTERP_FACTR7 = 127      # Just one hyperbolla + arctan(x) for an asymetric/distortion feature
INTERP_FACTR8 = 128      # Just one hyperbolla + x for asymetry + arctan(x) for an asymetric/distortion feature

INTERP_NONLI1 = 2000      # non-linear hyperbolla + x for asymetry + arctan(x) for an asymetric/distortion feature

INTERP_SVI000 = 2040      # Gatheral SVI model
INTERP_SVI100 = 2140      # Gatheral SVI model constrained 
INTERP_SVI110 = 2141      # Gatheral SVI model constrained to have a negative slope far left Better starting values
INTERP_SVI120 = 2121      # Gatheral SVI model constrained, with tight min region
INTERP_SVI001 = 2041      # Gatheral SVI model + arctan(x) for an asymetric/distortion feature
INTERP_SVI002 = 2042      # Gatheral SVI model + arctan(b2*x) for an asymetric/distortion feature
INTERP_SVI102 = 2142      # Gatheral SVI model + arctan(b2*x) for an asymetric/distortion feature

INTERP_3D_M2VOL = 3002  # 3D polynomial of order 2 on vol
INTERP_3D_M2VAR = 3004  # 3D polynomial of order 2 on variance

INTERP_3D_FGVGG = 3200  # Francois, Galarneau-Vincent, Gauthier, & Godin 2022 on IVolS

INTERP_3D_SVI00 = 4100  # 3D SVI +t + m*t + t**2
INTERP_3D_SVI01 = 4101  # 3D SVI +t + m*t + time-to-maturity slope


EXTRAP_LINEAR = 10       # works only for METHOD_STDR_EXTRAPIV
EXTRAP_GP3PTS = 20       # works only for METHOD_STDR_EXTRADEN
EXTRAP_GBETA2 = 21       # works only for METHOD_STDR_EXTRADEN
EXTRAP_ASYMPT = 30       # works only for METHOD_TLSM_EXTRAPIV
EXTRAP_GEV3PT = 40       # works only for METHOD_STDR_EXTRADEN

DENSITY_RANGE_DEFAULT = 0
DENSITY_RANGE_EXTENDD = 1
DENSITY_RANGE_KEPASIS = 2



def ols_wols(X: np.ndarray, y: np.ndarray, weights: np.ndarray=np.array([])) -> np.ndarray:
    if len(weights) > 0:
        W = np.diag(weights)
        beta = np.linalg.lstsq(W @ X, W @ y, rcond=-1)[0]
    else:
        beta = np.linalg.lstsq(X, y, rcond=-1)[0]
    return beta


def residuals(params, x, y, model_func, weights=np.array([]), constraints=[]):
    """
    General residuals function for least_squares.

    Parameters:
    - params: List of parameters to optimize.
    - x: Independent variable data.
    - y: Dependent variable data (target).
    - model_func: The non-linear model function to fit.

    Returns:
    - Residuals (difference between predicted and actual values).
    """

    penalty = 0

    if constraints:
        for constraint in constraints:
            penalty += constraint(*params)

    if penalty>0:
        pause = 1

    if len(weights) > 0:
        return weights**2 * (y - model_func(x, *params)) + penalty
    else:
        return y - model_func(x, *params) + penalty


# Define the non-linear function
def non_linear_func31(x, a0, a1, a2, b0):
    # return a0 + a1 * np.sqrt(b0 + b1 * ((x+b3) ** 2)) + a2 * x + a3 * np.arctan(b2 * (x+b4))
    return a0 + a1 * np.sqrt(1 + 1 * (x ** 2)) + a2 * x


def non_linear_SVI000(x, a0, a1, a2, b0, b1):
    # return a0 + a1 * np.sqrt(b0 + b1 * ((x+b3) ** 2)) + a2 * x + a3 * np.arctan(b2 * (x+b4))
    # return (a0 + a1 * (x-b0) + a2 * np.sqrt(b1 + (x-b0)**2) - a2 * np.sqrt(b1))
    # return (a0 + a1 * (x-b0) + a2 * np.sqrt(b1 + (x-b0)**2))
    
    # Now, use the original SVI model
    return a0 + a2 * (a1 * (x - b0) + np.sqrt(b1**2 + (x - b0) ** 2)) - a2*b1*np.sqrt(1-a1**2)


def SVI000_con_u(a0, a1, a2, b0, b1):
    # The basic constrainst are handled by the bounds on the parameters
    # a2 > 0
    # b1 > 0
    # -1 < a1 < 1
    raise("Not implemented yet")
    # Next, we need a0 + a2*b1*sqrt*1+a1**2 > =
    # condition = a0 + a2*b1*np.sqrt(1-a1**2)
    # penalty = max(0, -condition)

    # if penalty>0:
    #     penalty = 5*(1+penalty)**2
    # return penalty


def SVI002_con_u(a0, a1, a2, a3, b0, b1, b2):
    # The basic constrainst are handled by the bounds on the parameters
    # a2 > 0
    # b1 > 0
    # -1 < a1 < 1

    # Next, we need a0 + a2*b1*sqrt*1+a1**2 > =
    condition = a0 + a2*b1*np.sqrt(1+a1**2)
    penalty = max(0, -condition)

    if penalty>0:
        penalty = penalty*999999
    return penalty


def non_linear_SVI03D(x, a0, a1, a2, a3, a4, a5, b0, b1):
    # return a0 + a1 * np.sqrt(b0 + b1 * ((x+b3) ** 2)) + a2 * x + a3 * np.arctan(b2 * (x+b4))
    return (a0 + a1 * (x[:,0]-b0) + a2 * np.sqrt(b1 + (x[:,0]-b0)**2) - a2 * np.sqrt(b1)) + a3*x[:,1] + a4*x[:,1]*x[:,0] + a5*x[:,1]**2


def non_linear_SVI13D(x, a0, a1, a2, a3, a4, a5, b0, b1):
    # return a0 + a1 * np.sqrt(b0 + b1 * ((x+b3) ** 2)) + a2 * x + a3 * np.arctan(b2 * (x+b4))
    # return (a0 + a1 * (x[:,0]-b0) + a2 * np.sqrt(b1 + (x[:,0]-b0)**2) - a2 * np.sqrt(b1)) + a3*x[:,1] + a4*((1 - np.exp(- (np.multiply(x[:,0],x[:,0])))) * np.log(x[:,1]/5)) + a5*(np.exp(-np.sqrt(x[:,1]/0.25)))
    return (a0 + a1 * (x[:,0]-b0) + a2 * np.sqrt(b1 + (x[:,0]-b0)**2) - a2 * np.sqrt(b1)) + a3*x[:,1] + a4*(x[:,0] * np.log(x[:,1]/5)) + a5*(np.exp(-np.sqrt(x[:,1]/0.25)))


def non_linear_SVI001(x, a0, a1, a2, a3, b0, b1):
    # return a0 + a1 * np.sqrt(b0 + b1 * ((x+b3) ** 2)) + a2 * x + a3 * np.arctan(b2 * (x+b4))
    base  = non_linear_SVI000(x, a0, a1, a2, b0, b1)
    return (base + a3 * np.arctan(x-b0))


def non_linear_SVI002(x, a0, a1, a2, a3, b0, b1, b2):
    # return a0 + a1 * np.sqrt(b0 + b1 * ((x+b3) ** 2)) + a2 * x + a3 * np.arctan(b2 * (x+b4))
    base  = non_linear_SVI000(x, a0, a1, a2, b0, b1)
    return (base + a3 * np.arctan(b2*(x-b0)))


def _density_support(densityrange: Union[int, List[int]], K: np.ndarray) -> np.ndarray:
    if isinstance(densityrange, int):
        # Handle the case where densityrange is an integer
        if densityrange == DENSITY_RANGE_DEFAULT:
            localrange = np.array([min(K)*0.5, max(K)*1.5])
        elif densityrange == DENSITY_RANGE_EXTENDD:
            localrange = np.array([0.05, max(K)*2])
        # Add more cases as needed
        elif densityrange == DENSITY_RANGE_KEPASIS:
            localrange = np.array([min(K), max(K)])
        else:
            raise ValueError("Invalid densityrange integer value")
    elif isinstance(densityrange, list) and len(densityrange) == 2:
        # Handle the case where densityrange is a 2-element list
        localrange = np.array(densityrange)
    else:
        raise ValueError("Invalid densityrange format") 

    return localrange


def _interpolate(interp: int, x: np.ndarray, y: np.ndarray, newx: np.ndarray, weights: np.array=np.array([])) -> np.ndarray:
    if interp==INTERP_LINEAR:    # Linear interpolation
        # weights in linear interpolation are not used
        # do linear interpolation
        newy = np.interp(newx, x, y)
        # plt.plot(outputx[interpmask], outputy); plt.show()
        # pass
    elif (interp>10 and interp<20):   # Polynomial interpolation
        # Now, if we have weights, we need to adjust the estimation of the simple polynomial
        if len(weights)>0:
            # Fit a polynomial of order (interp-10) to the data
            p = np.poly1d(np.polyfit(x, y, interp-10, w=weights))
        else:
            p = np.poly1d(np.polyfit(x, y, interp-10))

        newy = p(newx)
        # plt.plot(outputx[interpmask], outputy[interpmask]); plt.show()
        # plt.plt()
        # pass
    elif (interp>99 and interp<2000):   # Affine Factor model interpolation
        # Fit a factor model with the SET#1
        if interp==INTERP_FACTR1:
            X = np.ones((len(x),2))
            X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
        
            Xout = np.ones((len(newx),2))
            Xout[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        elif interp==INTERP_FACTR2:
            # Fit a factor model with the SET#2
            X = np.ones((len(x),3))
            X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:,2] = np.sqrt(0.1+x**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
            
            Xout = np.ones((len(newx),3))
            Xout[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:,2] = np.sqrt(0.1+newx**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)

        elif interp==INTERP_FACTR3:
            # Fit a factor model with the SET#3
            X = np.ones((len(x),4))
            X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:,2] = np.sqrt(0.1+x**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
            X[:,3] = x
            
            Xout = np.ones((len(newx),4))
            Xout[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:,2] = np.sqrt(0.1+newx**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
            Xout[:,3] = newx

        elif interp==INTERP_FACTR31:
            # Fit a factor model with the SET#3
            X = np.ones((len(x),4))
            # find where y is at the minimum
            minindex = np.argmin(y)
            # find the value of x at the minimum
            minx = x[minindex]

            X[:,1] = np.sqrt(1+(x-minx)**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:,2] = np.sqrt(0.1+(x-minx)**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
            X[:,3] = x
            
            Xout = np.ones((len(newx),4))
            Xout[:,1] = np.sqrt(1+(newx-minx)**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:,2] = np.sqrt(0.1+(newx-minx)**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
            Xout[:,3] = newx


        elif interp==INTERP_FACTR4:
            # Fit a factor model with the SET#3
            X = np.ones((len(x),4))
            X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:,2] = np.sqrt(0.1+x**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
            X[:,3] = np.arctan(x)           # atan = np.arctan(m)
            
            Xout = np.ones((len(newx),4))
            Xout[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:,2] = np.sqrt(0.1+newx**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
            Xout[:,3] = np.arctan(newx)           # atan = np.arctan(m)


        elif interp==INTERP_FACTR5:
            # Fit a factor model with the SET#3
            X = np.ones((len(x),5))
            X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:,2] = np.sqrt(0.1+x**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
            X[:,3] = x
            X[:,4] = np.arctan(x)           # atan = np.arctan(m)
            
            Xout = np.ones((len(newx),5))
            Xout[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:,2] = np.sqrt(0.1+newx**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
            Xout[:,3] = newx
            Xout[:,4] = np.arctan(newx)           # atan = np.arctan(m)


        elif interp==INTERP_FACTR51:
            # Fit a factor model with the SET#3
            X = np.ones((len(x),7))
            X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:,2] = np.sqrt(1+2*x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:,3] = np.sqrt(1+0.5*x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:,4] = np.sqrt(1+10*x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:,5] = x
            X[:,6] = np.arctan(x)           # atan = np.arctan(m)

            
            Xout = np.ones((len(newx),7))
            Xout[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:,2] = np.sqrt(1+2*newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:,3] = np.sqrt(1+0.5*newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:,4] = np.sqrt(1+10*newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:,5] = newx
            Xout[:,6] = np.arctan(newx)           # atan = np.arctan(m)


        elif interp==INTERP_FACTR52:
            # Fit a factor model with the SET#3
            X = np.ones((len(x),12))
            X[:, 1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:, 2] = np.sqrt(1+2*x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:, 3] = np.sqrt(1+0.5*x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:, 4] = np.sqrt(1+10*x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:, 5] = np.sqrt(1+0.1*x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:, 6] = x
            X[:, 7] = np.arctan(x)           # atan = np.arctan(m)
            X[:, 8] = np.arctan(2*x)           # atan = np.arctan(m)
            X[:, 9] = np.arctan(0.5*x)           # atan = np.arctan(m)
            X[:,10] = np.arctan(10*x)           # atan = np.arctan(m)
            X[:,11] = np.arctan(0.1*x)           # atan = np.arctan(m)

            
            Xout = np.ones((len(newx),12))
            Xout[:, 1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:, 2] = np.sqrt(1+2*newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:, 3] = np.sqrt(1+0.5*newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:, 4] = np.sqrt(1+10*newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:, 5] = np.sqrt(1+0.1*newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:, 6] = newx
            Xout[:, 7] = np.arctan(newx)           # atan = np.arctan(m)
            Xout[:, 8] = np.arctan(2*newx)           # atan = np.arctan(m)
            Xout[:, 9] = np.arctan(0.5*newx)           # atan = np.arctan(m)
            Xout[:,10] = np.arctan(10*newx)           # atan = np.arctan(m)
            Xout[:,11] = np.arctan(0.1*newx)           # atan = np.arctan(m)


        elif interp==INTERP_FACTR6:
            # Fit a factor model with the SET#3
            X = np.ones((len(x),3))
            X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:,2] = x

            
            Xout = np.ones((len(newx),3))
            Xout[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:,2] = newx


        elif interp==INTERP_FACTR7:
            # Fit a factor model with the SET#3
            X = np.ones((len(x),3))
            X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:,2] = np.arctan(x)           # atan = np.arctan(m)

            
            Xout = np.ones((len(newx),3))
            Xout[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:,2] = np.arctan(newx)           # atan = np.arctan(m)


        elif interp==INTERP_FACTR8:
            # Fit a factor model with the SET#3
            X = np.ones((len(x),4))
            X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
            X[:,2] = x
            X[:,3] = np.arctan(x)           # atan = np.arctan(m)

            
            Xout = np.ones((len(newx),4))
            Xout[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
            Xout[:,2] = newx
            Xout[:,3] = np.arctan(newx)           # atan = np.arctan(m)

        else:
            raise ValueError("Interpolation method not recognized.")

        beta = ols_wols(X,y,weights)
        newy = np.matmul(Xout,beta)
        # pass

    elif (interp>1999 and interp<3000):     # NON-Affine Factor model interpolation
        
        selected_constraints = []

        if interp==INTERP_NONLI1:
            thefunction = non_linear_func31
            # Initial guess for the parameters

            initial_guess = [np.mean(y), 1, 0, 1]
            
        elif interp==INTERP_SVI000:
            thefunction = non_linear_SVI000
            # Initial guess for the parameters
            # non_linear_SVI000(x, a0, a1, a2, b0, b1):
            initial_guess = [np.min(y)**2, 0, 0.1, -0.1, 1]
            # return np.sqrt(a0 + a1 * (x-b0) + a2 * np.sqrt(b1 + (x-b0)**2) - a2 * np.sqrt(b1))

            # Define bounds: (lower_bounds, upper_bounds)
            #               cte,    slope/ratio  hypercurve  location,  curve
            lower_bounds = [      0,         -1,          0,   -np.inf,      0]  # a0 > 0, -0.1 < b0
            upper_bounds = [+np.inf,         +1,     np.inf,   +np.inf, np.inf]  # b0 < 0.1, rest unbounded


        elif interp==INTERP_SVI100:
            thefunction = non_linear_SVI000
            # Initial guess for the parameters
            # non_linear_SVI000(x, a0,    a1, a2,   b0, b1):
            initial_guess = [np.min(y)**2, 0, 1, -0.1, 1]
            # return np.sqrt(a0 + a1 * (x-b0) + a2 * np.sqrt(b1 + (x-b0)**2) - a2 * np.sqrt(b1))

            # Define bounds: (lower_bounds, upper_bounds)
            lower_bounds = [-np.inf, -1,       0,   -np.inf,      0]  # a0 > 0, -0.1 < b0
            upper_bounds = [+np.inf, +1,  np.inf,   +np.inf, np.inf]  # b0 < 0.1, rest unbounded
            
            selected_constraints = [SVI000_con_u]

        elif interp==INTERP_SVI120:
            thefunction = non_linear_SVI000
            # Initial guess for the parameters
            # non_linear_SVI000(x, a0,    a1, a2,   b0, b1):
            initial_guess = [np.min(y)**2, 0, 1, -0.1, 1]
            # return np.sqrt(a0 + a1 * (x-b0) + a2 * np.sqrt(b1 + (x-b0)**2) - a2 * np.sqrt(b1))
            # sort x on the order of y and put the result in xsorted
            xsorted = x[np.argsort(y)]
            # calculate how many is 10% of the data in x
            n = max(int(len(xsorted)*0.1),3)
            # find where y is at the minimum
            minindex = np.argmin(y)
            # find the value of x at the minimum
            minx = x[minindex]

            # Define bounds: (lower_bounds, upper_bounds)
            lower_bounds = [-np.inf, -1,       0, np.min(xsorted[:n]),      0]  # a0 > 0, -0.1 < b0
            upper_bounds = [+np.inf, +1,  np.inf, np.max(xsorted[:n]), np.inf]  # b0 < 0.1, rest unbounded
            
            initial_guess = [np.min(y)**2, 0, 1, minx, 1]
           
            selected_constraints = [SVI000_con_u]

        elif interp==INTERP_SVI110:
            thefunction = non_linear_SVI000
            # Initial guess for the parameters
            # non_linear_SVI000(x, a0,    a1, a2,   b0, b1):

            # find where y is at the minimum
            minindex = np.argmin(y)
            # find the value of x at the minimum
            minx = x[minindex]
            # I want to sort x and y according to x
            xsorted = x[np.argsort(x)]
            ysorted = y[np.argsort(x)]
            # find the slope of the curve at the far left and call it L
            L = min((ysorted[1]-ysorted[0])/(xsorted[1]-xsorted[0]),0)
            # find the slope of the curve at the far right and call it R
            R = max((ysorted[-1]-ysorted[-2])/(xsorted[-1]-xsorted[-2]),0)
            a2 = (R-L)/2
            a1 = a2 + L

            initial_guess = [np.min(y)**2 - a2*0.01*np.sqrt(1-a2**2), a1, a2, minx, 0.01]
            # return np.sqrt(a0 + a1 * (x-b0) + a2 * np.sqrt(b1 + (x-b0)**2) - a2 * np.sqrt(b1))

            # Define bounds: (lower_bounds, upper_bounds)
            lower_bounds = [-np.inf, -1,       0,   -np.inf,      0]  # a0 > 0, -0.1 < b0
            upper_bounds = [+np.inf, +1,  np.inf,   +np.inf, np.inf]  # b0 < 0.1, rest unbounded
            
            selected_constraints = [SVI000_con_u]

        elif interp==INTERP_SVI001:
            thefunction = non_linear_SVI001
            # Initial guess for the parameters
            # return np.sqrt(a0 + a1 * (x-b0) + a2 * np.sqrt(b1 + (x-b0)**2) - a2 * np.sqrt(b1+b0**2) + a3 * np.arctan(x-b0))
            #               [a0,           a1,  a2,  a3,   b0, b1, b2]
            initial_guess = [np.mean(y)**2, 0,   0.1,   0,  -0.1,  1]

            # Define bounds: (lower_bounds, upper_bounds)
            lower_bounds = [0      , -1,       0,       0, -np.inf,       0]  # a0 > 0, -0.1 < b0
            upper_bounds = [+np.inf, +1,  np.inf,  np.inf, +np.inf,  np.inf]  # b0 < 0.1, rest unbounded
            
        elif interp==INTERP_SVI002:
            thefunction = non_linear_SVI002
            # Initial guess for the parameters
            # non_linear_SVI000(x, a0, a1, a2, a3, b0, b1):
            #               [a0,            a1,  a2, a3,   b0, b1, b2]
            initial_guess = [np.min(y)**2,  0,  0.1,  0,  0.1,  1,  1]
            # return np.sqrt(a0           + a1 * (x-b0) + a2 * np.sqrt(b1 + (x-b0)**2) - a2 * np.sqrt(b1)) + a3 * np.arctan(b2*(x-b0))

            # Define bounds: (lower_bounds, upper_bounds)
            lower_bounds = [-np.inf, -1,       0, -np.inf, -np.inf,       0, -np.inf]  # a0 > 0, -0.1 < b0
            upper_bounds = [+np.inf, +1,  np.inf,  np.inf, +np.inf,  np.inf,  np.inf]  # b0 < 0.1, rest unbounded

        elif interp==INTERP_SVI102:
            thefunction = non_linear_SVI002
            # Initial guess for the parameters
            # non_linear_SVI000(x, a0, a1, a2, a3, b0, b1):
            #               [a0,            a1,  a2, a3,   b0, b1, b2]
            initial_guess = [np.min(y)**2,  0,  0.1,  0,  0.1,  1,  1]
            # return np.sqrt(a0           + a1 * (x-b0) + a2 * np.sqrt(b1 + (x-b0)**2) - a2 * np.sqrt(b1)) + a3 * np.arctan(b2*(x-b0))

            # Define bounds: (lower_bounds, upper_bounds)
            lower_bounds = [-np.inf, -1,       0, -np.inf, -np.inf,       0, -np.inf]  # a0 > 0, -0.1 < b0
            upper_bounds = [+np.inf, +1,  np.inf,  np.inf, +np.inf,  np.inf,  np.inf]  # b0 < 0.1, rest unbounded

            selected_constraints = [SVI002_con_u]

        # constraints = format_constraints(selected_constraints)

        params = least_squares(residuals, initial_guess, args=(x, y**2, thefunction), bounds=(lower_bounds, upper_bounds), method='trf').x
        if len(weights) > 0 or len(selected_constraints) > 0:
            # params = np.array([0.22, -.03, 0.30, -0.63, 0.73])
            resbeforeconstraints = residuals(params, x, y**2, thefunction, weights, selected_constraints)
            params = least_squares(residuals, params,        args=(x, y**2, thefunction, weights, selected_constraints), bounds=(lower_bounds, upper_bounds), method='trf').x
            resafterconstraints = residuals(params, x, y**2, thefunction, weights, selected_constraints)
            paramsini = least_squares(residuals, initial_guess,        args=(x, y**2, thefunction, weights, selected_constraints), bounds=(lower_bounds, upper_bounds), method='trf').x
            resafterconstraintsini = residuals(params, x, y**2, thefunction, weights, selected_constraints)
            if np.sum(resafterconstraintsini) < np.sum(resafterconstraints):
                params = paramsini
            # penalty = selected_constraints[0](*params)
            # if penalty>0:
            #     pause = 1
            pass

        # Predict new values
        newvar = thefunction(newx, *params)
        if np.any(newvar<0):
            # print('Negative Variance')
            newvar[newvar<0] = 0
        newy = np.sqrt(newvar)

        # if abs(params[1])>params[2]:
        #     pause = 1

        # print(params)

        # initresid = residuals(params, x, y**2, thefunction)

        # # Fit the curve
        # if selected_constraints:
        #     params = least_squares(residuals, params, args=(x, y**2, thefunction, weights, selected_constraints), bounds=(lower_bounds, upper_bounds), method='trf').x
        # else:
        #     params = least_squares(residuals, params, args=(x, y**2, thefunction, weights), bounds=(lower_bounds, upper_bounds), method='trf').x

        # secndresid = residuals(params, x, y**2, thefunction)

        # print(params)

        # if len(weights)>0 or len(selected_constraints)>0:
        #     # Predict new values
        #     print('Residuals:', np.sum(initresid**2), np.sum(secndresid**2))

    elif (interp>2999 and interp<4000):     # Affine SURFACE Factor model interpolation
        if interp==INTERP_3D_M2VOL:
            X = np.ones((len(x),6))
            X[:,1] = x[:,0]
            X[:,2] = x[:,0]**2
            X[:,3] = x[:,1]
            X[:,4] = x[:,1]**2
            X[:,5] = x[:,0]*x[:,1]
            
            Xnew = np.ones((len(newx),6))
            Xnew[:,1] = newx[:,0]
            Xnew[:,2] = newx[:,0]**2
            Xnew[:,3] = newx[:,1]
            Xnew[:,4] = newx[:,1]**2
            Xnew[:,5] = newx[:,0]*newx[:,1]
            

            # elif interp==INTERP_3D_M2VAR:
            #     X = np.ones((len(x),6))
            #     X[:,1] = x[:,0]
            #     X[:,2] = x[:,0]**2
            #     X[:,3] = x[:,1]
            #     X[:,4] = x[:,1]**2
            #     X[:,5] = x[:,0]*x[:,1]

            #     beta = np.linalg.lstsq(X,y**2,rcond=-1)[0]
                
            #     X = np.ones((len(newx),6))
            #     X[:,1] = newx[:,0]
            #     X[:,2] = newx[:,0]**2
            #     X[:,3] = newx[:,1]
            #     X[:,4] = newx[:,1]**2
            #     X[:,5] = newx[:,0]*newx[:,1]
                
            #     newy2 = np.matmul(X,beta)
            #     if np.any(newy2<0):
            #         # print('Negative Variance')
            #         newy2[newy2<0] = 0

            #     newy = np.sqrt(newy2)

        elif interp==INTERP_3D_FGVGG:
            X = np.ones((len(x),5))                                                                                 # constant
            X[:,1] = np.exp(-np.sqrt(x[:,1]/0.25))                                                                  # Time-to-Maturity Slope
            X[:,2] = ( (np.exp(2*x[:,0])-1) / (np.exp(2*x[:,0])+1))*(x[:,0]<0) + x[:,0]*(x[:,0]>=0)                   # Moneyness Slope
            X[:,3] = ((1 - np.exp(- (np.multiply(x[:,0],x[:,0])))) * np.log(x[:,1]/5))                   # Smile Attenuation
            left = x[:,0]*(x[:,0]<0)
            X[:,4] = ((1 - np.exp((3*left)**3)) * np.log(x[:,1]/5))*(x[:,0]<0)                             # Smirk

            Xnew = np.ones((len(newx),5))
            Xnew[:,1] = np.exp(-np.sqrt(newx[:,1]/0.25))
            Xnew[:,2] = ( (np.exp(2*newx[:,0])-1) / (np.exp(2*newx[:,0])+1))*(newx[:,0]<0) + newx[:,0]*(newx[:,0]>=0) 
            Xnew[:,3] = ((1 - np.exp(- (np.multiply(newx[:,0],newx[:,0])))) *  np.log(newx[:,1]/5))
            left = newx[:,0]*(x[:,0]<0)
            Xnew[:,4] = (1 - np.exp((3*left)**3)) * np.log(newx[:,1]/5)*(newx[:,0]<0)

        beta = np.linalg.lstsq(X,y,rcond=-1)[0]

        newy = np.matmul(Xnew,beta)  

        # end of AFFINE SURFACE Factor model interpolation

    elif (interp>3999 and interp<5000):    # NON-Affine SURFACE Factor model interpolation
        if interp==INTERP_3D_SVI00:
            # a0 LEVEL + a1 SLOPE + a2 SVI + a3 t + a4 t*m + a5 t^2                 
            # (a0 + a1 * (x[:,0]-b0) + a2 * np.sqrt(b1 + (x[:,0]-b0)**2) - a2 * np.sqrt(b1+b0**2)) + a3*x[:,1] + a4*x[:,1]*x[:,0] + a5*x[:,1]**2
            # Initial guess for the parameters
            #               [           a0, a1, a2, a3, a4, a5, b0, b1]:
            initial_guess = [np.mean(y)**2,  0,  1,  0,  0,  0,  0,  1]
            # 

            # Define bounds: (lower_bounds, upper_bounds)
            #              [    a0,      a1,      a2,      a3,      a4,      a5,      b0,      b1]
            lower_bounds = [0     , -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,       0]  # a0 > 0, -0.1 < b0
            upper_bounds = [np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf]  # b0 < 0.1, rest unbounded

            interpfunction = non_linear_SVI03D


        elif interp==INTERP_3D_SVI01:
            # 
            # a0 LEVEL + a1 SLOPE + a2 SVI + a3 t + a4 Smile_Attenuation + a5 Time-to-Maturity Slope
            # a0 + a1 * (x[:,0]-b0) + a2 * np.sqrt(b1 + (x[:,0]-b0)**2) - a2 * np.sqrt(b1)) + a3*x[:,1] + a4*(x[:,0] * np.log(x[:,1]/5)) + a5*(np.exp(-np.sqrt(x[:,1]/0.25))
            # Initial guess for the parameters
            #               [           a0, a1, a2, a3, a4, a5, b0, b1]:
            initial_guess = [np.mean(y)**2,  0,  1,  0,  0,  0,  0,  1]
            # return np.sqrt(a0 + a1 * (x-b0) + a2 * np.sqrt(b1 + (x-b0)**2) - a2 * np.sqrt(b1))

            # Define bounds: (lower_bounds, upper_bounds)
            lower_bounds = [0     , -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,       0]  # a0 > 0, -0.1 < b0
            upper_bounds = [np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf]  # b0 < 0.1, rest unbounded

            interpfunction = non_linear_SVI13D

            # # Fit the curve
            # params = least_squares(residuals, initial_guess, args=(x, y**2, non_linear_SVI13D), bounds=(lower_bounds, upper_bounds), method='trf').x
            
            # # Predict new values
            # newy2 = non_linear_SVI13D(newx, *params)
            # if np.any(newy2<0):
            #     # print('Negative Variance')
            #     newy2[newy2<0] = 0

            # newy = np.sqrt(newy2)


        # Fit the curve
        params = least_squares(residuals, initial_guess, args=(x, y**2, interpfunction), bounds=(lower_bounds, upper_bounds), method='trf').x
        
        # Predict new values
        newy2 = interpfunction(newx, *params)
        if np.any(newy2<0):
            # print('Negative Variance')
            newy2[newy2<0] = 0
        newy = np.sqrt(newy2)

        # end of NON-AFFINE SURFACE Factor model interpolation

    return newy


def _scale(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    fmin = np.min(f)
    if fmin<0:
        f = f - fmin
        # This is a way to prevent negative probabilities without creating "breaks" in the density
    cumulative = np.trapz(f,x)
    f = f/cumulative
    return f


def getfit(x: np.ndarray, y: np.ndarray, interp: int=INTERP_POLYM3, newx: np.ndarray=np.array([]), weights: np.ndarray=np.array([])) -> np.ndarray:
    if newx.size==0:
        return _interpolate(interp, x, y, x, weights)
    else:
        return _interpolate(interp, x, y, newx, weights)


def getfitextrapolated(x: np.ndarray, y: np.ndarray, newx: np.ndarray, interp: int=INTERP_POLYM3, weights: np.ndarray=np.array([])) -> np.ndarray:
    return _interpolate(interp, x, y, newx, weights)


def getrnd(K: np.ndarray, V: np.ndarray, S: float, rf: float, t: float, interp: int=INTERP_POLYM3, method: int=METHOD_STDR_EXTRAPIV, extrap: int=EXTRAP_LINEAR, 
           densityrange: Union[int, List[int]]=DENSITY_RANGE_DEFAULT, nbpoints:int = 10000, fittingweights:np.ndarray=np.array([])) -> tuple:
    """
    For now, the function only deals with one maturity.
    Parameters:
        K (np.ndarray): The strikes, provided in a vector
        V (np.ndarray): The matching IVs for OTM+ATM calls and puts
        S (float): the spot price of the underlying
        rf (float): the risk-free rate in continuous time, decimal
        t (float): the time to maturity in years
        interp (int): The interpolation method to use. Default is INTERP_POLYM3. Possible values are INTERP_LINEAR, INTERP_POLYM#, INTERP_FACTR#, INTERP_NONLI1
        method (int): The method to use to extrapolate the RND.
            METHOD_STDR_EXTRAPIV: Extrapolate the IV as constants from the extremities of the support. Then smooth the density at the junction.
            METHOD_STDR_EXTRADEN: Extrapolate the density using a Generalized Pareto (TODO: extend to another distribution).
            METHOD_TLSM_EXTRAPIV: Extrapolate the IV in the time-scaled log-moneyness space using the asymptotes of the model.
        
        Once we have the IVs, we can get the put prices, and then the convexity of the put prices to get the rnd.
        rnd = exp(rf*t) * d^2P/dK^2

        at the output, the RND is scale to have a total probability of 1.


    Returns:
        outputx,                    : The support of the RND
        outputy,                    : The smoothed IV curve
        _scale(outputx, outputf),   : The RND, scaled to have a total probability of 1
        outputf,                    : The RND, unscaled to compare with the scaled version
        thetaIV,                    : The parameters of the model when fitting the IV curve
        thetaRND                    : The parameters of the model when fitting the RND tails, if applicable

    Suppose extrap = EXTRAP_GP3PTS, method = METHOD_STDR_EXTRADEN, and densityrange = DENSITY_RANGE_EXTENDD
    """
    N = len(K)  # Determine the size N based on the length of K
    K = np.array(K)  # Convert K to a NumPy array
    V = np.array(V)  # Convert V to a NumPy array
    if K.shape != (N,):
        raise ValueError(f"Input K must be of size ({N},), but got shape {K.shape}")
    if V.shape != (N,):
        raise ValueError(f"Input V must be of same size ({N},), but got shape {V.shape}")
    
    # get the support for the density
    localrange = _density_support(densityrange, K)

    # Now, create the output vectors
    outputx = np.linspace(localrange[0], localrange[1], nbpoints)
    outputy = np.zeros_like(outputx)
    outputf = np.zeros_like(outputx)
    
    # get the interpolatable portion, and two tails of extrapolation
    maskinterpol = (outputx>=np.min(K)) & (outputx<=np.max(K))
    masklefttail = outputx<min(K)
    maskrightail = outputx>max(K)

    # get the interpolatable portion of the IV curve
    if method==METHOD_STDR_EXTRAPIV:
        # interpolate first
        outputy[maskinterpol] = _interpolate(interp, K, V, outputx[maskinterpol], fittingweights)

        # Now extrapolate as a constant
        outputy[masklefttail] = outputy[maskinterpol][0]
        outputy[    maskrightail] = outputy[maskinterpol][-1]
        # plt.plot(outputx, outputy); plt.show()
        
        # now get Black-Scholes-Merton put prices.
        p = bls('p', S=S, K=outputx, t=t, r=rf, sigma=outputy, return_as='np')
        # plt.plot(outputx[interpmask], p[interpmask]); plt.show()
        
        # get convexity
        outputf = np.exp(rf * t) * np.gradient(np.gradient(p, outputx, edge_order=2), outputx, edge_order=2)
        # plt.plot(outputx, outputf); plt.show(); #plt.xlim(5,15), 
        
        # now, we need to smooth the disconnection points and smooth them out
        mask = np.array([True, False, False, False, True])
        indexhole = np.argmin(masklefttail)
        x = outputx[(indexhole - 2):(indexhole + 2 + 1)][mask]
        y = outputf[(indexhole - 2):(indexhole + 2 + 1)][mask]
        outputf[indexhole-1:indexhole+1+1] = np.interp(outputx[(indexhole - 2):(indexhole + 2 + 1)][~mask], x, y)
        indexhole = np.argmax(    maskrightail)
        x = outputx[(indexhole - 3):(indexhole + 1 + 1)][mask]
        y = outputf[(indexhole - 3):(indexhole + 1 + 1)][mask]
        outputf[indexhole-2:indexhole+1] = np.interp(outputx[(indexhole - 3):(indexhole+1+1)][~mask], x, y)

        # plt.plot(outputx, outputf); plt.show()
                
        # pass
        
        
    elif method==METHOD_STDR_EXTRADEN:
        # interpolate first
        outputy[maskinterpol] = _interpolate(interp, K, V, outputx[maskinterpol], fittingweights)
        # now get Black-Scholes-Merton put prices.
        p = bls('p', S=S, K=outputx[maskinterpol], t=t, r=rf, sigma=outputy[maskinterpol], return_as='np')
        # plt.plot(outputx[interpmask], p[interpmask]); plt.show()
        # get convexity
        outputf[maskinterpol] = np.exp(rf * t) * np.gradient(np.gradient(p, outputx[maskinterpol], edge_order=2), outputx[maskinterpol], edge_order=2)
        # plt.plot(outputx[interpmask], outputf[interpmask]); plt.show()
        
        npts = 2
        xlefttailfit = outputx[maskinterpol][0:npts][::-1]
        refpoint = outputx[maskinterpol][npts]
        xlefttailfit = -1*(xlefttailfit - refpoint)
        ylefttailfit = outputf[maskinterpol][0:npts][::-1]
        xlefttailext = -1*(outputx[masklefttail][::-1] - refpoint)

        xrighttailfit = outputx[maskinterpol][-npts:]
        refpoint = outputx[maskinterpol][-npts-1]
        xrighttailfit = xrighttailfit - refpoint
        yrighttailfit = outputf[maskinterpol][-npts:]
        xrighttailext = outputx[    maskrightail] - refpoint

        # Generalized Pareto and Generalized Beta 2.
        if extrap==EXTRAP_GP3PTS:
            outputf = opt.fittails(outputx, maskinterpol, masklefttail,     maskrightail, outputf, opt.F_GENPARETO)

            # TODO 2024-11-14: This one here fits the 3 parameters of the Generalized Pareto distribution to the 3 points of the current density
            # We start with the left tail. Since Generalized Pareto is a distribution for the right tail, we need to flip the left tail.
            # We fit the left tail to the leftmost 3 points of the density

            # *****DEBUG: check whether we have the right data
            # *****x3pts = outputx[interpmask][0:3]
            # *****y3pts = outputf[interpmask][0:3]
            # *****plt.plot(outputx[interpmask], outputf[interpmask])
            # *****plt.scatter(x3pts, y3pts)
            # *****plt.show()
            # *****VERIFIED: we do have the right data.
            # *****Now we flip the left tail, We need the values of x 
            # *****x3ptsflipped = x3pts[::-1]
            # *****y3ptsflipped = y3pts
            # *****plt.scatter(x3ptsflipped, y3ptsflipped)
            # *****plt.show()
            # *****VERIFIED: Now we have what we need to fit the left tail.
            # npts = 2
            # xlefttailfit = outputx[interpmask][0:npts][::-1]
            # refpoint = outputx[interpmask][npts]
            # xlefttailfit = -1*(xlefttailfit - refpoint)
            # ylefttailfit = outputf[interpmask][0:npts][::-1]
            # xlefttailext = -1*(outputx[extlftmask][::-1] - refpoint)

            # xrighttailfit = outputx[interpmask][-npts:]
            # refpoint = outputx[interpmask][-npts-1]
            # xrighttailfit = xrighttailfit - refpoint
            # yrighttailfit = outputf[interpmask][-npts:]
            # xrighttailext = outputx[extrgtmask] - refpoint

            # thetaleft = opt._fittail(xlefttailfit, ylefttailfit, opt.evalgenpareto)

            outputf = opt.fittails(outputx, maskinterpol, masklefttail,     maskrightail, outputf, opt.F_GENPARETO)

            # DEBUG: check whether we have the right output
            # fout = genpareto.pdf(xlefttailfit, c=thetaleft[0], loc=0, scale=thetaleft[1])
            # plt.scatter(outputx[interpmask][0:npts], ylefttailfit)
            # plt.scatter(outputx[interpmask][0:npts], fout, marker='o')
            # plt.scatter(xlefttailfit, fout, marker='o')

            # extendf = genpareto.pdf(xlefttailext, c=thetaleft[0], loc=0, scale=thetaleft[1])
            # extendx = outputx[extlftmask]

            # plt.scatter(xlefttailext[:5], extendf[:5])
            # plt.scatter(xlefttailext, extendf)
            # plt.scatter(extendx[-5:], extendf[-5:])

            # plt.scatter(xlefttailfit, ylefttailfit, s=10)
            # plt.show()

            # plt.scatter(xrighttailfit, yrighttailfit)
            # plt.show()


            # outputf[extlftmask] = (genpareto.pdf(xlefttailext, c=thetaleft[0], loc=0, scale=thetaleft[1]))
            # outputf[extlftmask] = outputf[extlftmask][::-1]
            # # DEBUG: check whether we have the right output
            # plt.plot(outputx[interpmask], outputf[interpmask])
            # plt.plot(outputx[extlftmask], outputf[extlftmask])
            # plt.show()

            # Now we fit the right tail to the rightmost 2 points of the density
            # xrighttailfit = outputx[interpmask][-npts:]
            # refpoint = outputx[interpmask][-npts-1]
            # xrighttailfit = xrighttailfit - refpoint
            # yrighttailfit = outputf[interpmask][-npts:]
            # xrighttailext = outputx[extrgtmask] - refpoint



            # thetarigt = opt._fittail(xrighttailfit, yrighttailfit, opt.evalgenpareto)
            # outputf[extrgtmask] = genpareto.pdf(xrighttailext, c=thetarigt[0], loc=0, scale=thetarigt[1])
            # plt.plot(outputx[interpmask], outputf[interpmask])
            # # plt.plot(outputx[extrgtmask], outputf[extrgtmask])
            # plt.plot(outputx[extlftmask][::-1], outputf[extlftmask]); 
            # plt.show()
            
            pass
        elif extrap==EXTRAP_GBETA2:
            raise ValueError("Not implemented yet")
            # thetaleft = opt._fittail(outputx[interpmask][2]-outputx[interpmask][0:2][::-1], outputf[interpmask][0:2][::-1], opt.evalgenpareto)
            # outputf[extlftmask] = (genpareto.pdf(outputx[interpmask][2]-outputx[extlftmask][::-1], thetaleft[0], loc=thetaleft[1]))
            # thetarigt = opt._fittail(outputx[interpmask][-3:-1], outputf[interpmask][-3:-1], opt.evalgenpareto)
            # outputf[extrgtmask] = (genpareto.pdf(outputx[extrgtmask], thetarigt[0], loc=thetarigt[1]))
            # plt.plot(outputx[interpmask], outputf[interpmask])
            # plt.plot(outputx[extrgtmask], outputf[extrgtmask])
            # plt.plot(outputx[extlftmask][::-1], outputf[extlftmask]); plt.show()
        elif extrap==EXTRAP_GEV3PT:
            outputf = opt.fittails(outputx, maskinterpol, masklefttail,     maskrightail, outputf, opt.F_GENEXTREME)


        else:
            raise ValueError("This extrapolation method is not valid for this method.")
        
        
        pass
    

    elif method==METHOD_TLSM_EXTRAPIV:
        # convert to TSLM frame
        m = np.multiply(1/np.sqrt(t), np.log(S*np.exp(rf*t)/K))
        newm = np.multiply(1/np.sqrt(t), np.log(S*np.exp(rf*t)/outputx))
        # plt.plot(m,V)#; plt.show()
        outputy = _interpolate(interp, m, V, newm, fittingweights)
        # plt.plot(outputx, outputy); plt.show()
        p = bls('p', S=S, K=outputx, t=t, r=rf, sigma=outputy, return_as='np')
        # get convexity
        outputf = np.exp(rf * t) * np.gradient(np.gradient(p, outputx, edge_order=2), outputx, edge_order=2)

        # Now check the integral here
        cumulative = simpson(outputf, outputx)
        # now print the cumulative value for the SVI model
        print(f"Integral of the density with SVI extrapolation: {cumulative}")
        # plt.plot(outputx, outputf); plt.show()
        pass
    else:
        raise ValueError("Invalid method. Use the recommended methods") 
    


    # return outputx, outputy, _scale(outputx, outputf)
    return outputx, outputy, _scale(outputx, outputf), outputf
    pause=1
    
    
def getmoments(x: np.ndarray, f: np.ndarray)->tuple:
    # Matlab equivalent
    # expected_value = trapz(x, x .* y);
    # variance = trapz(x, ((x - expected_value).^2) .* y);
    # third_moment = trapz(x, ((x - expected_value).^3) .* y);
    # skewness = third_moment / variance^(3/2);   
    # fourth_moment = trapz(x, ((x - expected_value).^4) .* y);
    # excess_kurtosis = fourth_moment / variance^2 - 3;
    
    func = lambda x_val: x_val * np.interp(x_val, x, f)
    expected_value, error = quad(func, np.min(x), np.max(x))
    
    func = lambda x_val: (x_val - expected_value)**2 * np.interp(x_val, x, f)
    variance, error = quad(func, np.min(x), np.max(x))
    
    func = lambda x_val: (x_val - expected_value)**3 * np.interp(x_val, x, f)
    thirdmoment, error = quad(func, np.min(x), np.max(x))
    skewness = thirdmoment/variance**(3/2)

    func = lambda x_val: (x_val - expected_value)**4 * np.interp(x_val, x, f)
    fourthmoment, error = quad(func, np.min(x), np.max(x))
    excesskurtosis = fourthmoment/variance**(2) - 3

    return expected_value, variance, skewness, excesskurtosis
    # pass
    
    
    