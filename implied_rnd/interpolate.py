import numpy as np
from scipy.optimize import least_squares


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

INTERP_SVI000 = 40      # Gatheral SVI model
INTERP_SVI001 = 41      # Gatheral SVI model + arctan(x) for an asymetric/distortion feature
INTERP_SVI002 = 42      # Gatheral SVI model + arctan(b2*x) for an asymetric/distortion feature


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

    elif interp==INTERP_NONLI1:
        # 
        # Initial guess for the parameters

        initial_guess = [np.mean(y), 1, 0, 1]
        # return a0 + a1 * np.sqrt(1 + b0 * (x ** 2)) + a2 * x
        # Fit the curve
        params, pcov = curve_fit(non_linear_func31, x, y, p0=initial_guess, maxfev=50000)
        
        # Predict new values
        newy = non_linear_func31(newx, *params)
        
    elif interp==INTERP_SVI000:
        # 
        # Initial guess for the parameters
        # non_linear_SVI000(x, a0, a1, a2, b0, b1):
        initial_guess = [np.mean(y)**2, 1, 1, 0, 1]
        # return np.sqrt(a0 + a1 * (x-b0) + a2 * np.sqrt(b1 + (x-b0)**2) - a2 * np.sqrt(b1))

        # Define bounds: (lower_bounds, upper_bounds)
        lower_bounds = [0     , -np.inf, -np.inf, -0.1, 0]  # a0 > 0, -0.1 < b0
        upper_bounds = [np.inf,  np.inf,  np.inf,  0.1,  np.inf]  # b0 < 0.1, rest unbounded

        # debug

        # y = non_linear_SVI000(x, .20, 1, 1, 0, 1)
        # e = residuals([.20, 1, 1, 0, 1], x, y, non_linear_SVI000)

        # Fit the curve
        params = least_squares(residuals, initial_guess, args=(x, y**2, non_linear_SVI000), bounds=(lower_bounds, upper_bounds), method='trf').x
        
        # Predict new values
        newy = np.sqrt(non_linear_SVI000(newx, *params))

    elif interp==INTERP_SVI001:
        # 
        # Initial guess for the parameters
        # non_linear_SVI000(x, a0, a1, a2, a3, b0, b1):
        initial_guess = [np.mean(y)**2, 1, 1, 0, 0, 1]
        # return np.sqrt(a0 + a1 * (x-b0) + a2 * np.sqrt(b1 + (x-b0)**2) - a2 * np.sqrt(b1))

        # Define bounds: (lower_bounds, upper_bounds)
        lower_bounds = [0     , -np.inf, -np.inf, -np.inf, -0.1, 0]  # a0 > 0, -0.1 < b0
        upper_bounds = [np.inf,  np.inf,  np.inf,  np.inf,  0.1,  np.inf]  # b0 < 0.1, rest unbounded

        # debug

        # y = non_linear_SVI000(x, .20, 1, 1, 0, 1)
        # e = residuals([.20, 1, 1, 0, 1], x, y, non_linear_SVI000)

        # Fit the curve
        params = least_squares(residuals, initial_guess, args=(x, y**2, non_linear_SVI001), bounds=(lower_bounds, upper_bounds), method='trf').x
        
        # Predict new values
        newy = np.sqrt(non_linear_SVI001(newx, *params))
        
    elif interp==INTERP_SVI002:
        # 
        # Initial guess for the parameters
        # non_linear_SVI000(x, a0, a1, a2, a3, b0, b1):
        initial_guess = [np.mean(y)**2, 1, 1, 0, 0, 1, 1]
        # return np.sqrt(a0 + a1 * (x-b0) + a2 * np.sqrt(b1 + (x-b0)**2) - a2 * np.sqrt(b1))

        # Define bounds: (lower_bounds, upper_bounds)
        lower_bounds = [0     , -np.inf, -np.inf, -np.inf, -0.1, 0,       -np.inf]  # a0 > 0, -0.1 < b0
        upper_bounds = [np.inf,  np.inf,  np.inf,  np.inf,  0.1,  np.inf,  np.inf]  # b0 < 0.1, rest unbounded

        # debug

        # y = non_linear_SVI000(x, .20, 1, 1, 0, 1)
        # e = residuals([.20, 1, 1, 0, 1], x, y, non_linear_SVI000)

        # Fit the curve
        params = least_squares(residuals, initial_guess, args=(x, y**2, non_linear_SVI002), bounds=(lower_bounds, upper_bounds), method='trf').x
        
        # Predict new values
        newy = np.sqrt(non_linear_SVI002(newx, *params))
        
    return newy
