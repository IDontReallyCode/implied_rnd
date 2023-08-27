"""
    You need to provide two arrays: strikes, and IVs from OTM/ATM calls and puts
    
    There will be a default method applied, but you can select the method you want.
"""

import numpy as np
from numba import njit


STDR = 0    # Standard representation with strikes on the x axis
TSLM = 1    # Use the time-scaled log-moneyness




