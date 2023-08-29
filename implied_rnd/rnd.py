"""
    You need to provide two arrays: strikes, and IVs from OTM/ATM calls and puts
    
    There will be a default method applied, but you can select the method you want.
"""

import numpy as np
from numba import njit
from typing import Union, List


STDR = 0    # Standard representation with strikes on the x axis
TSLM = 1    # Use the time-scaled log-moneyness

INTERP_LINEAR = 0
INTERP_P2 = 2
INTERP_P3 = 3
INTERP_P4 = 4
INTERP_P5 = 5

EXTRAP_PRE_LINEAR = 0

DENSITY_RANGE_DEFAULT = 0
DENSITY_RANGE_EXTENDD = 1

def getrnd(K: np.ndarray, V: np.ndarray, F: float=1, interp: int=INTERP_P3, extrap: int=EXTRAP_PRE_LINEAR, 
           densityrange: Union[int, List[int]]=DENSITY_RANGE_DEFAULT) -> np.ndarray:
    """
    Assign the input variable K to a NumPy array of size (N,).

    Parameters:
        K (np.ndarray): The input variable to assign.

    Returns:
        np.ndarray: A NumPy array of size (N,).
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
            localrange = np.array([0, max(K)*2])
        # Add more cases as needed
        else:
            raise ValueError("Invalid densityrange integer value")
    elif isinstance(densityrange, list) and len(densityrange) == 2:
        # Handle the case where densityrange is a 2-element list
        localrange = np.array(densityrange)
    else:
        raise ValueError("Invalid densityrange format") 

    pause=1