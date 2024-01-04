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
ticker = 'TSLA'

def main():
    
    rawrf = GET_ALL_RF()
    rfeod = np.array([dtd[1] for dtd in rawrf])
    rfbdr = np.array([a[2] for a in rawrf],dtype=float)
    
    # Get a list of all folders in the root path
    # folders = [f for f in glob.glob(mainfolder + "**/", recursive=True)]
    folders = next(os.walk(mainfolder))[1]

    # Sort the list of folders
    folders.sort()
    
    Ncuts = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15]
    nbcuts = len(Ncuts)
    # Take the expected value as the metric
    metricM1method1 = np.zeros((len(folders),nbcuts))
    metricM1method2 = np.zeros((len(folders),nbcuts))
    metricM1method31 = np.zeros((len(folders),nbcuts))
    metricM1method32 = np.zeros((len(folders),nbcuts))
    metricM1method33 = np.zeros((len(folders),nbcuts))
    metricM1method34 = np.zeros((len(folders),nbcuts))
    metricM1method35 = np.zeros((len(folders),nbcuts))
    
    metricM2method1 = np.zeros((len(folders),nbcuts))
    metricM2method2 = np.zeros((len(folders),nbcuts))
    metricM2method31 = np.zeros((len(folders),nbcuts))
    metricM2method32 = np.zeros((len(folders),nbcuts))
    metricM2method33 = np.zeros((len(folders),nbcuts))
    metricM2method34 = np.zeros((len(folders),nbcuts))
    metricM2method35 = np.zeros((len(folders),nbcuts))
    
    metricM3method1 = np.zeros((len(folders),nbcuts))
    metricM3method2 = np.zeros((len(folders),nbcuts))
    metricM3method31 = np.zeros((len(folders),nbcuts))
    metricM3method32 = np.zeros((len(folders),nbcuts))
    metricM3method33 = np.zeros((len(folders),nbcuts))
    metricM3method34 = np.zeros((len(folders),nbcuts))
    metricM3method35 = np.zeros((len(folders),nbcuts))
    
    metricM4method1 = np.zeros((len(folders),nbcuts))
    metricM4method2 = np.zeros((len(folders),nbcuts))
    metricM4method31 = np.zeros((len(folders),nbcuts))
    metricM4method32 = np.zeros((len(folders),nbcuts))
    metricM4method33 = np.zeros((len(folders),nbcuts))
    metricM4method34 = np.zeros((len(folders),nbcuts))
    metricM4method35 = np.zeros((len(folders),nbcuts))
    
    
    
    iday = 0
    for thisdate in folders:
        print(thisdate)
        date_object = datetime.datetime.strptime(thisdate, '%Y%m%d').date()
        thisfile = mainfolder + thisdate + '/' + ticker + thisdate + '.pbz2'
        try:
            f = bz2.BZ2File(thisfile, 'r')
            OC = pickle.load(f)
        except:
            continue
        
        S = OC['underlyingPrice']
        try:
            r = rfbdr[rfeod==date_object][0]/100
        except:
            pass
        
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
            try:
                V[i] = OC['putExpDateMap'][daylist[where]][strike][0]['volatility']/100       
            except:
                continue
            # C[i] = OC['callExpDateMap'][daylist[where]][strike][0]['volatility']/100       
        
        x0, y0, f0 = rnd.getrnd(K, V, S=S, rf=r, t=T, interp=rnd.INTERP_POLYM3, densityrange=rnd.DENSITY_RANGE_DEFAULT)
        x1, y1, f1 = rnd.getrnd(K, V, S=S, rf=r, t=T, method=rnd.METHOD_STDR_EXTRADEN, interp=rnd.INTERP_POLYM3, extrap=rnd.EXTRAP_GPARTO, densityrange=rnd.DENSITY_RANGE_DEFAULT)
        x21, y2, f21 = rnd.getrnd(K, V, S=S, rf=r, t=T, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR1, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_DEFAULT)
        x22, y2, f22 = rnd.getrnd(K, V, S=S, rf=r, t=T, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR2, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_DEFAULT)
        x23, y2, f23 = rnd.getrnd(K, V, S=S, rf=r, t=T, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR6, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_DEFAULT)
        x24, y2, f24 = rnd.getrnd(K, V, S=S, rf=r, t=T, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR7, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_DEFAULT)
        x25, y2, f25 = rnd.getrnd(K, V, S=S, rf=r, t=T, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR8, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_DEFAULT)
        
        # plt.plot(x0,f0)
        # plt.plot(x1,f1)
        # plt.plot(x2,f2)
        # plt.show()

        left = K<S
        rigt = K>S
        
        # plt.scatter(K,left)
        # plt.scatter(K,rigt)
        # plt.show()
        
        # get reference moments
        M01, M02, M03, M04 = rnd.getmoments(x0, f0)
        M11, M12, M13, M14 = rnd.getmoments(x1, f1)
        M211, M221, M231, M241 = rnd.getmoments(x21, f21)
        M212, M222, M232, M242 = rnd.getmoments(x22, f22)
        M213, M223, M233, M243 = rnd.getmoments(x23, f23)
        M214, M224, M234, M244 = rnd.getmoments(x24, f24)
        M215, M225, M235, M245 = rnd.getmoments(x25, f25)
        for icut, Ncut in enumerate(Ncuts):

            # cut X% on both sides
            # Ncut = 0.02
            Nleft = int(len(K)*Ncut)
            Nrigt = int(len(K)*(1-Ncut))
            print(Nleft)
            
            xc0, yc0, fc0 = rnd.getrnd(K[Nleft:Nrigt], V[Nleft:Nrigt], S=S, rf=r, t=T, interp=rnd.INTERP_POLYM3, densityrange=rnd.DENSITY_RANGE_DEFAULT)
            xc1, yc1, fc1 = rnd.getrnd(K[Nleft:Nrigt], V[Nleft:Nrigt], S=S, rf=r, t=T, method=rnd.METHOD_STDR_EXTRADEN, interp=rnd.INTERP_POLYM3, extrap=rnd.EXTRAP_GPARTO, densityrange=rnd.DENSITY_RANGE_DEFAULT)
            xc21, yc21, fc21 = rnd.getrnd(K[Nleft:Nrigt], V[Nleft:Nrigt], S=S, rf=r, t=T, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR1, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_DEFAULT)
            xc22, yc22, fc22 = rnd.getrnd(K[Nleft:Nrigt], V[Nleft:Nrigt], S=S, rf=r, t=T, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR2, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_DEFAULT)
            xc23, yc23, fc23 = rnd.getrnd(K[Nleft:Nrigt], V[Nleft:Nrigt], S=S, rf=r, t=T, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR6, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_DEFAULT)
            xc24, yc24, fc24 = rnd.getrnd(K[Nleft:Nrigt], V[Nleft:Nrigt], S=S, rf=r, t=T, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR7, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_DEFAULT)
            xc25, yc25, fc25 = rnd.getrnd(K[Nleft:Nrigt], V[Nleft:Nrigt], S=S, rf=r, t=T, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR8, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_DEFAULT)
            
            # moments after the cuts
            Mc01, Mc02, Mc03, Mc04 = rnd.getmoments(xc0, fc0)
            Mc11, Mc12, Mc13, Mc14 = rnd.getmoments(xc1, fc1)
            Mc211, Mc221, Mc231, Mc241 = rnd.getmoments(xc21, fc21)
            Mc212, Mc222, Mc232, Mc242 = rnd.getmoments(xc22, fc22)
            Mc213, Mc223, Mc233, Mc243 = rnd.getmoments(xc23, fc23)
            Mc214, Mc224, Mc234, Mc244 = rnd.getmoments(xc24, fc24)
            Mc215, Mc225, Mc235, Mc245 = rnd.getmoments(xc25, fc25)
            
            metricM1method1[iday,icut] = (Mc01-M01)/M01
            metricM1method2[iday,icut] = (Mc11-M11)/M11
            metricM1method31[iday,icut] = (Mc211-M211)/M211
            metricM1method32[iday,icut] = (Mc212-M212)/M212
            metricM1method33[iday,icut] = (Mc213-M213)/M213
            metricM1method34[iday,icut] = (Mc214-M214)/M214
            metricM1method35[iday,icut] = (Mc215-M215)/M215

            metricM2method1[iday,icut] = (Mc02-M02)/M02
            metricM2method2[iday,icut] = (Mc12-M12)/M12
            metricM2method31[iday,icut] = (Mc221-M221)/M221
            metricM2method32[iday,icut] = (Mc222-M222)/M222
            metricM2method33[iday,icut] = (Mc223-M223)/M223
            metricM2method34[iday,icut] = (Mc224-M224)/M224
            metricM2method35[iday,icut] = (Mc225-M225)/M225

            metricM3method1[iday,icut] = (Mc03-M03)/M03
            metricM3method2[iday,icut] = (Mc13-M13)/M13
            metricM3method31[iday,icut] = (Mc231-M231)/M231
            metricM3method32[iday,icut] = (Mc232-M232)/M232
            metricM3method33[iday,icut] = (Mc233-M233)/M233
            metricM3method34[iday,icut] = (Mc234-M234)/M234
            metricM3method35[iday,icut] = (Mc235-M235)/M235

            metricM4method1[iday,icut] = (Mc04-M04)/M04
            metricM4method2[iday,icut] = (Mc14-M14)/M14
            metricM4method31[iday,icut] = (Mc241-M241)/M241
            metricM4method32[iday,icut] = (Mc242-M242)/M242
            metricM4method33[iday,icut] = (Mc243-M243)/M243
            metricM4method34[iday,icut] = (Mc244-M244)/M244
            metricM4method35[iday,icut] = (Mc245-M245)/M245
            

            # print(metriccompare)
            # if iday>100:
            #     iday=iday
            #     break

            pass
        iday+=1
    
    pass

    plt.plot(Ncuts, (np.mean(metricM1method1,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM1method2,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM1method31,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM1method32,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM1method33,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM1method34,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM1method35,axis=0)))
    plt.show()
    
    plt.plot(Ncuts, (np.mean(metricM2method1,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM2method2,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM2method31,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM2method32,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM2method33,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM2method34,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM2method35,axis=0)))
    plt.show()
    
    plt.plot(Ncuts, (np.mean(metricM3method1,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM3method2,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM3method31,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM3method32,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM3method33,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM3method34,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM3method35,axis=0)))
    plt.show()
    
    plt.plot(Ncuts, (np.mean(metricM4method1,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM4method2,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM4method31,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM4method32,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM4method33,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM4method34,axis=0)))
    plt.plot(Ncuts, (np.mean(metricM4method35,axis=0)))
    plt.show()
    
    
    pass




if __name__ == '__main__':
    main()
