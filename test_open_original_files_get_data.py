import pickle
import bz2
import os
# import glob
from sqlstuff import GET_ALL_RF
import numpy as np
import datetime
import implied_rnd as rnd
import matplotlib.pyplot as plt

mainfolder = 'C:/Users/p_h_d/Dropbox/00_INVESTING/Data/'

def main():
    
    rawrf = GET_ALL_RF()
    rfeod = np.array([dtd[1] for dtd in rawrf])
    rfbdr = np.array([a[2] for a in rawrf],dtype=float)
    
    # Get a list of all folders in the root path
    # folders = [f for f in glob.glob(mainfolder + "**/", recursive=True)]
    folders = next(os.walk(mainfolder))[1]

    # Sort the list of folders
    folders.sort()
    
    for thisdate in folders:
        print(thisdate)
        date_object = datetime.datetime.strptime(thisdate, '%Y%m%d').date()
        thisfile = mainfolder + thisdate + '/AMZN' + thisdate + '.pbz2'
        try:
            f = bz2.BZ2File(thisfile, 'r')
            OC = pickle.load(f)
        except:
            continue
        
        S = OC['underlyingPrice']
        r = rfbdr[rfeod==date_object][0]/100
        
        # find the maturities and extract the DTE from the keys
        dtes = np.array([int(key.split(":")[1]) for key in OC['putExpDateMap'].keys()])
        where = np.argmin(np.abs(dtes-30))
        T = float(dtes[where])/365
        daylist = list(OC['putExpDateMap'].keys())
        
        # Now get the strikes
        strikeys = list(OC['putExpDateMap'][daylist[where]].keys())
        n = len(strikeys)
        K = np.zeros((n,))
        V = np.zeros((n,))
        # C = np.zeros((n,))
        
        for i, strike in enumerate(strikeys):
            K[i] = float(strike)
            V[i] = OC['putExpDateMap'][daylist[where]][strike][0]['volatility']/100       
            # C[i] = OC['callExpDateMap'][daylist[where]][strike][0]['volatility']/100       
        
        x0, y0, f0 = rnd.getrnd(K, V, S=S, rf=r, t=T, interp=rnd.INTERP_POLYM3, densityrange=rnd.DENSITY_RANGE_DEFAULT)
        x1, y1, f1 = rnd.getrnd(K, V, S=S, rf=r, t=T, method=rnd.METHOD_STDR_EXTRADEN, interp=rnd.INTERP_POLYM3, extrap=rnd.EXTRAP_GPARTO, densityrange=rnd.DENSITY_RANGE_DEFAULT)
        x2, y2, f2 = rnd.getrnd(K, V, S=S, rf=r, t=T, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR1, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_DEFAULT)
        
        # plt.plot(x0,f0)
        # plt.plot(x1,f1)
        # plt.plot(x2,f2)
        # plt.show()

        left = K<S
        rigt = K>S
        
        # plt.scatter(K,left)
        # plt.scatter(K,rigt)
        # plt.show()
        
        

        pass
    
    
    pass




if __name__ == '__main__':
    main()
