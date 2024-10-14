"""
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

import matplotlib.pyplot as plt

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

INTERP_FACTR1 = 21      # Just one hyperbolla
INTERP_FACTR2 = 22      # Two different hyperbollas
INTERP_FACTR3 = 23      # Two hyperbollas + x for an asymetric feature
INTERP_FACTR31 = 231    # Two hyperbollas (centered at min value) + x for an asymetric feature
INTERP_FACTR4 = 24      # Two hyperbollas + arctan(x) for an asymetric/distortion feature
INTERP_FACTR5 = 25      # Two hyperbollas + x for asymetry + arctan(x) for an asymetric/distortion feature
INTERP_FACTR51 = 251    # four hyperbollas + x for asymetry + arctan(x) for an asymetric/distortion feature
INTERP_FACTR52 = 252    # four hyperbollas + x for asymetry + three arctan(x) for an asymetric/distortion feature

INTERP_FACTR6 = 26      # Just one hyperbolla + x for asymetry 
INTERP_FACTR7 = 27      # Just one hyperbolla + arctan(x) for an asymetric/distortion feature
INTERP_FACTR8 = 28      # Just one hyperbolla + x for asymetry + arctan(x) for an asymetric/distortion feature

INTERP_NONLI1 = 31      # non-linear hyperbolla + x for asymetry + arctan(x) for an asymetric/distortion feature

EXTRAP_LINEAR = 10       # works only for METHOD_STDR_EXTRAPIV
EXTRAP_GPARTO = 20       # works only for METHOD_STDR_EXTRADEN
EXTRAP_GBETA2 = 21       # works only for METHOD_STDR_EXTRADEN
EXTRAP_ASYMPT = 30       # works only for METHOD_TLSM_EXTRAPIV

DENSITY_RANGE_DEFAULT = 0
DENSITY_RANGE_EXTENDD = 1


# Define the non-linear function
def non_linear_func31(x, a0, a1, a2, b0):
    # return a0 + a1 * np.sqrt(b0 + b1 * ((x+b3) ** 2)) + a2 * x + a3 * np.arctan(b2 * (x+b4))
    return a0 + a1 * np.sqrt(1 + 1 * (x ** 2)) + a2 * x



def _interpolate(interp: int, x: np.ndarray, y: np.ndarray, newx: np.ndarray) -> np.ndarray:
    if interp==INTERP_LINEAR:
        # do linear interpolation
        newy = np.interp(newx, x, y)
        # plt.plot(outputx[interpmask], outputy); plt.show()
        # pass
    elif (interp>10 and interp<20):
        #
        p = np.poly1d(np.polyfit(x, y, interp-10))

        newy = p(newx)
        # plt.plot(outputx[interpmask], outputy[interpmask]); plt.show()
        # plt.plt()
        # pass
    elif interp==21:
        # Fit a factor model with the SET#1
        X = np.ones((len(x),2))
        X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
        # X[:,2] = np.sqrt(0.1+x**2)-np.sqrt(0.1)    # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        # X[:,3] = np.arctan(x)           # atan = np.arctan(m)
        # X[:,4] = np.sin(x)              # m-shape
        # plt.plot(x, X[:,1]); plt.show()
        beta = np.linalg.lstsq(X,y,rcond=-1)[0]
        
        X = np.ones((len(newx),2))
        X[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        # X[:,2] = np.sqrt(0.1+newx**2)-np.sqrt(0.1)    # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        # X[:,3] = np.arctan(newx)           # atan = np.arctan(m)
        # X[:,4] = np.sin(newx)              # m-shape
        
        newy = np.matmul(X,beta)
        # pass
    elif interp==22:
        # Fit a factor model with the SET#2
        X = np.ones((len(x),3))
        X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,2] = np.sqrt(0.1+x**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        # X[:,3] = np.arctan(x)           # atan = np.arctan(m)
        # X[:,4] = np.sin(x)              # m-shape
        # plt.plot(x, X[:,1]); plt.show()
        beta = np.linalg.lstsq(X,y,rcond=-1)[0]
        
        X = np.ones((len(newx),3))
        X[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,2] = np.sqrt(0.1+newx**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        # X[:,3] = np.arctan(newx)           # atan = np.arctan(m)
        # X[:,4] = np.sin(newx)              # m-shape
        
        newy = np.matmul(X,beta)
    elif interp==INTERP_FACTR3:
        # Fit a factor model with the SET#3
        X = np.ones((len(x),4))
        X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,2] = np.sqrt(0.1+x**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        X[:,3] = x
        # X[:,3] = np.arctan(x)           # atan = np.arctan(m)
        # X[:,4] = np.sin(x)              # m-shape
        # plt.plot(x, X[:,1]); plt.show()
        beta = np.linalg.lstsq(X,y,rcond=-1)[0]
        
        X = np.ones((len(newx),4))
        X[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,2] = np.sqrt(0.1+newx**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        X[:,3] = newx
        # X[:,3] = np.arctan(newx)           # atan = np.arctan(m)
        # X[:,4] = np.sin(newx)              # m-shape
        
        newy = np.matmul(X,beta)        
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
        # X[:,3] = np.arctan(x)           # atan = np.arctan(m)
        # X[:,4] = np.sin(x)              # m-shape
        # plt.plot(x, X[:,1]); plt.show()
        beta = np.linalg.lstsq(X,y,rcond=-1)[0]
        
        X = np.ones((len(newx),4))
        X[:,1] = np.sqrt(1+(newx-minx)**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,2] = np.sqrt(0.1+(newx-minx)**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        X[:,3] = newx
        # X[:,3] = np.arctan(newx)           # atan = np.arctan(m)
        # X[:,4] = np.sin(newx)              # m-shape
        
        newy = np.matmul(X,beta)        
    elif interp==INTERP_FACTR4:
        # Fit a factor model with the SET#3
        X = np.ones((len(x),4))
        X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,2] = np.sqrt(0.1+x**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        # X[:,3] = x
        X[:,3] = np.arctan(x)           # atan = np.arctan(m)
        # X[:,4] = np.sin(x)              # m-shape
        # plt.plot(x, X[:,1]); plt.show()
        beta = np.linalg.lstsq(X,y,rcond=-1)[0]
        
        X = np.ones((len(newx),4))
        X[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,2] = np.sqrt(0.1+newx**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        # X[:,3] = newx
        X[:,3] = np.arctan(newx)           # atan = np.arctan(m)
        # X[:,4] = np.sin(newx)              # m-shape
        
        newy = np.matmul(X,beta)        
    elif interp==INTERP_FACTR5:
        # Fit a factor model with the SET#3
        X = np.ones((len(x),5))
        X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,2] = np.sqrt(0.1+x**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        X[:,3] = x
        X[:,4] = np.arctan(x)           # atan = np.arctan(m)
        # X[:,4] = np.sin(x)              # m-shape
        # plt.plot(x, X[:,1]); plt.show()
        beta = np.linalg.lstsq(X,y,rcond=-1)[0]
        
        X = np.ones((len(newx),5))
        X[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,2] = np.sqrt(0.1+newx**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        X[:,3] = newx
        X[:,4] = np.arctan(newx)           # atan = np.arctan(m)
        # X[:,4] = np.sin(newx)              # m-shape
        
        newy = np.matmul(X,beta)        

    elif interp==INTERP_FACTR51:
        # Fit a factor model with the SET#3
        X = np.ones((len(x),7))
        X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,2] = np.sqrt(1+2*x**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,3] = np.sqrt(1+0.5*x**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,4] = np.sqrt(1+10*x**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,5] = x
        X[:,6] = np.arctan(x)           # atan = np.arctan(m)
        # X[:,4] = np.sin(x)              # m-shape
        # plt.plot(x, X[:,1]); plt.show()
        beta = np.linalg.lstsq(X,y,rcond=-1)[0]
        
        X = np.ones((len(newx),7))
        X[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,2] = np.sqrt(1+2*newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,3] = np.sqrt(1+0.5*newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,4] = np.sqrt(1+10*newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:,5] = newx
        X[:,6] = np.arctan(newx)           # atan = np.arctan(m)
        # X[:,4] = np.sin(newx)              # m-shape
        
        newy = np.matmul(X,beta)        

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
        # X[:,4] = np.sin(x)              # m-shape
        # plt.plot(x, X[:,1]); plt.show()
        beta = np.linalg.lstsq(X,y,rcond=-1)[0]
        
        X = np.ones((len(newx),12))
        X[:, 1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:, 2] = np.sqrt(1+2*newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:, 3] = np.sqrt(1+0.5*newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:, 4] = np.sqrt(1+10*newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:, 5] = np.sqrt(1+0.1*newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        X[:, 6] = newx
        X[:, 7] = np.arctan(newx)           # atan = np.arctan(m)
        X[:, 8] = np.arctan(2*newx)           # atan = np.arctan(m)
        X[:, 9] = np.arctan(0.5*newx)           # atan = np.arctan(m)
        X[:,10] = np.arctan(10*newx)           # atan = np.arctan(m)
        X[:,11] = np.arctan(0.1*newx)           # atan = np.arctan(m)
        # X[:,4] = np.sin(newx)              # m-shape
        
        newy = np.matmul(X,beta)        

    elif interp==26:
        # Fit a factor model with the SET#3
        X = np.ones((len(x),3))
        X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
        # X[:,2] = np.sqrt(0.1+x**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        X[:,2] = x
        # X[:,4] = np.arctan(x)           # atan = np.arctan(m)
        # X[:,4] = np.sin(x)              # m-shape
        # plt.plot(x, X[:,1]); plt.show()
        beta = np.linalg.lstsq(X,y,rcond=-1)[0]
        
        X = np.ones((len(newx),3))
        X[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        # X[:,2] = np.sqrt(0.1+newx**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        X[:,2] = newx
        # X[:,4] = np.arctan(newx)           # atan = np.arctan(m)
        # X[:,4] = np.sin(newx)              # m-shape
        
        newy = np.matmul(X,beta)        

    elif interp==27:
        # Fit a factor model with the SET#3
        X = np.ones((len(x),3))
        X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
        # X[:,2] = np.sqrt(0.1+x**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        # X[:,3] = x
        X[:,2] = np.arctan(x)           # atan = np.arctan(m)
        # X[:,4] = np.sin(x)              # m-shape
        # plt.plot(x, X[:,1]); plt.show()
        beta = np.linalg.lstsq(X,y,rcond=-1)[0]
        
        X = np.ones((len(newx),3))
        X[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        # X[:,2] = np.sqrt(0.1+newx**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        # X[:,3] = newx
        X[:,2] = np.arctan(newx)           # atan = np.arctan(m)
        # X[:,4] = np.sin(newx)              # m-shape
        
        newy = np.matmul(X,beta)        

    elif interp==28:
        # Fit a factor model with the SET#3
        X = np.ones((len(x),4))
        X[:,1] = np.sqrt(1+x**2)-1      # symmetric-smile :: Hyperbolla b=1
        # X[:,2] = np.sqrt(0.1+x**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        X[:,2] = x
        X[:,3] = np.arctan(x)           # atan = np.arctan(m)
        # X[:,4] = np.sin(x)              # m-shape
        # plt.plot(x, X[:,1]); plt.show()
        beta = np.linalg.lstsq(X,y,rcond=-1)[0]
        
        X = np.ones((len(newx),4))
        X[:,1] = np.sqrt(1+newx**2)-1      # symmetric-smile :: Hyperbolla b=1
        # X[:,2] = np.sqrt(0.1+newx**2)-np.sqrt(0.1)      # symmetric-smile :: Hyperbolla b=sqrt(0.1)
        X[:,2] = newx
        X[:,3] = np.arctan(newx)           # atan = np.arctan(m)
        # X[:,4] = np.sin(newx)              # m-shape
        
        newy = np.matmul(X,beta)        

    elif interp==31:
        # 
        # Initial guess for the parameters

        initial_guess = [np.mean(y), 1, 0, 1]
        # return a0 + a1 * np.sqrt(1 + b0 * (x ** 2)) + a2 * x
        # Fit the curve
        params, pcov = curve_fit(non_linear_func31, x, y, p0=initial_guess, maxfev=50000)
        
        # Predict new values
        newy = non_linear_func31(newx, *params)
        
    return newy


def _scale(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    fmin = np.min(f)
    if fmin<0:
        f = f - fmin
        # This is a way to prevent negative probabilities without creating "breaks" in the density
    cumulative = np.trapz(f,x)
    f = f/cumulative
    return f


def getfit(x: np.ndarray, y: np.ndarray, interp: int=INTERP_POLYM3) -> np.ndarray:
    return _interpolate(interp, x, y, x)


def getfitextrapolated(x: np.ndarray, y: np.ndarray, newx: np.ndarray, interp: int=INTERP_POLYM3) -> np.ndarray:
    return _interpolate(interp, x, y, newx)


def getrnd(K: np.ndarray, V: np.ndarray, S: float, rf: float, t: float, interp: int=INTERP_POLYM3, method: int=METHOD_STDR_EXTRAPIV, extrap: int=EXTRAP_LINEAR, 
           densityrange: Union[int, List[int]]=DENSITY_RANGE_DEFAULT, nbpoints:int = 10000) -> tuple:
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
        np.ndarray: Two NumPy arrays of size (nbpoints,) one for underlying values, and one for density.

    Suppose extrap = EXTRAP_GPARTO, method = METHOD_STDR_EXTRADEN, and densityrange = DENSITY_RANGE_EXTENDD
    """
    N = len(K)  # Determine the size N based on the length of K
    K = np.array(K)  # Convert K to a NumPy array
    V = np.array(V)  # Convert V to a NumPy array
    if K.shape != (N,):
        raise ValueError(f"Input K must be of size ({N},), but got shape {K.shape}")
    if V.shape != (N,):
        raise ValueError(f"Input V must be of same size ({N},), but got shape {V.shape}")
    
    if isinstance(densityrange, int):
        # Handle the case where densityrange is an integer
        if densityrange == DENSITY_RANGE_DEFAULT:
            localrange = np.array([min(K)*0.5, max(K)*1.5])
        elif densityrange == DENSITY_RANGE_EXTENDD:
            localrange = np.array([0.05, max(K)*2])
        # Add more cases as needed
        else:
            raise ValueError("Invalid densityrange integer value")
    elif isinstance(densityrange, list) and len(densityrange) == 2:
        # Handle the case where densityrange is a 2-element list
        localrange = np.array(densityrange)
    else:
        raise ValueError("Invalid densityrange format") 

    # Now, create the output support
    outputx = np.linspace(localrange[0], localrange[1], nbpoints)
    outputy = np.zeros_like(outputx)
    outputf = np.zeros_like(outputx)
    
    # get the interpolatable portion, and two sides of extrapolation
    interpmask = (outputx>=np.min(K)) & (outputx<=np.max(K))
    extlftmask = outputx<min(K)
    extrgtmask = outputx>max(K)


    if method==METHOD_STDR_EXTRAPIV:
        # interpolate first
        outputy[interpmask] = _interpolate(interp, K, V, outputx[interpmask])

        # Now extrapolate as a constant
        outputy[extlftmask] = outputy[interpmask][0]
        outputy[extrgtmask] = outputy[interpmask][-1]
        # plt.plot(outputx, outputy); plt.show()
        
        # now get Black-Scholes-Merton put prices.
        p = bls('p', S=S, K=outputx, t=t, r=rf, sigma=outputy, return_as='np')
        # plt.plot(outputx[interpmask], p[interpmask]); plt.show()
        
        # get convexity
        outputf = np.exp(rf * t) * np.gradient(np.gradient(p, outputx, edge_order=2), outputx, edge_order=2)
        # plt.plot(outputx, outputf); plt.show(); #plt.xlim(5,15), 
        
        # now, we need to smooth the disconnection points and smooth them out
        mask = np.array([True, False, False, False, True])
        indexhole = np.argmin(extlftmask)
        x = outputx[(indexhole - 2):(indexhole + 2 + 1)][mask]
        y = outputf[(indexhole - 2):(indexhole + 2 + 1)][mask]
        outputf[indexhole-1:indexhole+1+1] = np.interp(outputx[(indexhole - 2):(indexhole + 2 + 1)][~mask], x, y)
        indexhole = np.argmax(extrgtmask)
        x = outputx[(indexhole - 3):(indexhole + 1 + 1)][mask]
        y = outputf[(indexhole - 3):(indexhole + 1 + 1)][mask]
        outputf[indexhole-2:indexhole+1] = np.interp(outputx[(indexhole - 3):(indexhole+1+1)][~mask], x, y)

        # plt.plot(outputx, outputf); plt.show()
                
        # pass
        
        
    elif method==METHOD_STDR_EXTRADEN:
        # interpolate first
        outputy[interpmask] = _interpolate(interp, K, V, outputx[interpmask])
        # now get Black-Scholes-Merton put prices.
        p = bls('p', S=S, K=outputx[interpmask], t=t, r=rf, sigma=outputy[interpmask], return_as='np')
        # plt.plot(outputx[interpmask], p[interpmask]); plt.show()
        # get convexity
        outputf[interpmask] = np.exp(rf * t) * np.gradient(np.gradient(p, outputx[interpmask], edge_order=2), outputx[interpmask], edge_order=2)
        # plt.plot(outputx[interpmask], outputf[interpmask]); plt.show()
        
        # Generalized Pareto and Generalized Beta 2.
        if extrap==EXTRAP_GPARTO:
            # We need to have a simple function that will optimize the different parameters of the Generalized Pareto density to match 2 or 3 points of the current density
            thetaleft = opt._fittail(outputx[interpmask][2]-outputx[interpmask][0:2][::-1], outputf[interpmask][0:2][::-1], opt.evalgenpareto)
            outputf[extlftmask] = (genpareto.pdf(outputx[interpmask][2]-outputx[extlftmask][::-1], thetaleft[0], loc=thetaleft[1]))[::-1]
            thetarigt = opt._fittail(outputx[interpmask][-3:-1], outputf[interpmask][-3:-1], opt.evalgenpareto)
            outputf[extrgtmask] = genpareto.pdf(outputx[extrgtmask], thetarigt[0], loc=thetarigt[1])
            # plt.plot(outputx[interpmask], outputf[interpmask])
            # plt.plot(outputx[extrgtmask], outputf[extrgtmask])
            # plt.plot(outputx[extlftmask][::-1], outputf[extlftmask]); plt.show()
            
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
            

        else:
            raise ValueError("This extrapolation method is not valid for this method.")
        
        
        pass
    

    elif method==METHOD_TLSM_EXTRAPIV:
        # convert to TSLM frame
        m = np.multiply(1/np.sqrt(t), np.log(S*np.exp(rf*t)/K))
        newm = np.multiply(1/np.sqrt(t), np.log(S*np.exp(rf*t)/outputx))
        # plt.plot(m,V)#; plt.show()
        outputy = _interpolate(interp, m, V, newm)
        # plt.plot(outputx, outputy); plt.show()
        p = bls('p', S=S, K=outputx, t=t, r=rf, sigma=outputy, return_as='np')
        # get convexity
        outputf = np.exp(rf * t) * np.gradient(np.gradient(p, outputx, edge_order=2), outputx, edge_order=2)
        # plt.plot(outputx, outputf); plt.show()
        pass
    else:
        raise ValueError("Invalid method. Use the recommended methods") 
    


    return outputx, outputy, _scale(outputx, outputf)
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
    
    
    