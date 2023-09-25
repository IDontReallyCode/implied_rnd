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

import matplotlib.pyplot as plt

METHOD_STDR_EXTRAPIV = 0    # Standard support with extrapolation of IV
METHOD_STDR_EXTRADEN = 1    # Standard support with extrapolation of density
METHOD_TLSM_EXTRAPIV = 2    # Time-Scaled-Log-Moneyness support with extrapolation of IV using asymptotes

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
INTERP_FACTR1 = 21

EXTRAP_LINEAR = 10       # works only for METHOD_STDR_EXTRAPIV
EXTRAP_GPARTO = 20       # works only for METHOD_STDR_EXTRADEN
EXTRAP_GBETA2 = 21       # works only for METHOD_STDR_EXTRADEN
EXTRAP_ASYMPT = 30       # works only for METHOD_TLSM_EXTRAPIV

DENSITY_RANGE_DEFAULT = 0
DENSITY_RANGE_EXTENDD = 1


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
        
    return newy


def _scale(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    cumulative = np.trapz(f,x)
    f = f/cumulative
    return f


def getrnd(K: np.ndarray, V: np.ndarray, S: float, rf: float, t: float, method: int=METHOD_STDR_EXTRAPIV, interp: int=INTERP_POLYM3, extrap: int=EXTRAP_LINEAR, 
           densityrange: Union[int, List[int]]=DENSITY_RANGE_DEFAULT, nbpoints:int = 10000) -> tuple:
    """
    Parameters:
        K (np.ndarray): The strikes.
        V (np.ndarray): The matching IVs for OTM+ATM calls and puts.
        F (float): The forward price of the underlying asset if the Time-Scaled-Log-Moneyness frame is used
        support (int): Use the standard frame, or the Time-Scaled-Log-Moneyness frame
        interp (int): Method used to interpolate values
        extrap (int): Method used to extrapolate values
        densityrange (int, [lower, upper]): range for the values for which we want a density
        nbpoints (int): how many points should be used when approximating the density

    Returns:
        np.ndarray: Two NumPy arrays of size (nbpoints,) one for underlying values, and one for density.
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