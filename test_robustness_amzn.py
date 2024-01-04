"""
    It appears this data has issues with the spot price.


    The objective here is to test the robustness of each method, using a regular stock that has lots of strikes and a wide range of log-moneyness.
    
    For one method:
    - We use the method to get the RND and then use numerical integration to get the first 4 moments, and consider those as the true values.
    - We test robustness on the left tail first
        - We remove 10% of moneyness, apply the method, get the first 4 moments. Calculate the % change.
        - Redo removing 10% more.
        - repeat until removed 30% of the left strikes.
        Repeat for right tail
        Repeat removing on both sides
        
    for each method, we calculate the % effect of removing n strikes on mean, vari, skew, kurt, on left, right, and both
    We try to remove 50 strikes. If we can't, then we drop this
    
"""

import os
import random
import pickle
import numpy as np
import implied_rnd as rnd
import matplotlib.pyplot as plt
from scipy import stats
import bz2

RootPath = 'C:/Users/p_h_d/Dropbox/00_INVESTING/Data/'

def main():
    dir_path = "..\\IV_SURFACE_FACTORS_LASSO\\cleaned_bidaskiv"
    filename = "AMZN.pkl"
    file_path = os.path.join(dir_path, filename)
    # Load the content of the file
    with open(file_path, 'rb') as f:
        content = pickle.load(f)
        
    # Get all keys from the dictionary
    all_keys = list(content.keys())
    ndays = len(all_keys)
    
    """
        Here, we are looking at what is wrong.
        I either have all strikes above or below the spot, which makes no sense for, let's say, AMZN
        So, here, I'll open one of the original file and look at the spot.
    """
    FileNameDated = RootPath + '20210507/AMZN20210507.pbz2'
    f = bz2.BZ2File(FileNameDated, 'r')
    OCcheck = pickle.load(f)
    
    """
        OK, so, the problem is that the "cleaned bid-ask spread is missing a lot of strikes"
        I am not sure why, though. I will need to redo- the cleaning
    """
    
    matuty = 30
    inventoryofstrikes = np.zeros((ndays,2))

    for ikey, thiskey in enumerate(all_keys):
        S = content[thiskey]['spot']
        rf = content[thiskey]['rf']
        

        # Get the numpy array
        array = np.array(content[thiskey]['dte'])
        array = np.sort(array)
        
        # Find the indices of the values just below and above maturity
        index_below = np.max(np.where(array < matuty))
        index_above = np.min(np.where(array > matuty))
        # print(array[index_below])
        # print(array[index_above])
        
        # Find the index of the element in the array that is closest to the value
        countbelow = (array[array==array[index_below]]).size
        countabove = (array[array==array[index_above]]).size
        
        mask = (array == array[index_above])
        K = content[thiskey]['strikes'][mask]
        t = array[index_above]/365
        m = np.multiply(1/np.sqrt(t), np.log(S*np.exp(rf*t)/K))
        leftmask = m>=0
        righmask = m<=0
        try:
            aboveratio = sum(leftmask)/sum(righmask)
        except:
            aboveratio = 99999999

        mask = (array == array[index_below])
        K = content[thiskey]['strikes'][mask]
        t = array[index_below]/365
        m = np.multiply(1/np.sqrt(t), np.log(S*np.exp(rf*t)/K))
        leftmask = m>=0
        righmask = m<=0
        try:
            belowratio = sum(leftmask)/sum(righmask)
        except:
            belowratio = 99999999
        
        aboveratio = np.abs(aboveratio-1)
        belowratio = np.abs(belowratio-1)
        
        if aboveratio<belowratio:
            mask = (array == array[index_above])
        else:
            mask = (array == array[index_below])
        K = content[thiskey]['strikes'][mask]
        m = np.multiply(1/np.sqrt(t), np.log(S*np.exp(rf*t)/K))
        leftmask = m>=0
        righmask = m<=0
            
        
        
        # index = (np.abs(array - matuty)).argmin()
        
        # Create a mask for the elements in the array that are closest to the value
        # mask = (array == array[index])
        
        # print(f"Mask for key {thiskey}: {mask}")
        V = (content[thiskey]['ask_iv'][mask] + content[thiskey]['bid_iv'][mask])/2
        
        # m = np.multiply(1/np.sqrt(t), np.log(S*np.exp(rf*t)/K))
        
        number_check = K.shape[0]
        
        leftmask = m>=0
        righmask = m<=0
        
        inventoryofstrikes[ikey,0] = int(sum(leftmask))
        inventoryofstrikes[ikey,1] = int(sum(righmask))
        
        
        

        print("")
        # if number_check>5:
        #     x0, y0, f0 = rnd.getrnd(K, V, S=S, rf=rf, t=t, interp=rnd.INTERP_POLYM3, densityrange=rnd.DENSITY_RANGE_EXTENDD)
        #     x1, y1, f1 = rnd.getrnd(K, V, S=S, rf=rf, t=t, method=rnd.METHOD_STDR_EXTRADEN, interp=rnd.INTERP_POLYM3, extrap=rnd.EXTRAP_GPARTO, densityrange=rnd.DENSITY_RANGE_EXTENDD)
        #     x2, y2, f2 = rnd.getrnd(K, V, S=S, rf=rf, t=t, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR1, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_EXTENDD)

        #     pass
            
    # print(count)
    pass
   
    
#### __name__ MAIN()
if __name__ == '__main__':
    # SET = int(sys.argv[1])
    main()

    

