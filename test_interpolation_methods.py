import implied_rnd as rnd
import numpy as np
import matplotlib.pyplot as plt

K = np.array([3,4,5,6,7,8,9,10,11,12,13,14,15])
V = np.array([.7256,.724,.723,.722,.721,.720,.720,.720,.7202, .7204, .7206, .7208, .721])

# rnd.getrnd(K, V, S=10, rf=0.0, t=0.25, densityrange=rnd.DENSITY_RANGE_EXTENDD, interp=rnd.INTERP_LINEAR)
# rnd.getrnd(K, V, S=10, rf=0.0, t=0.25, densityrange=rnd.DENSITY_RANGE_EXTENDD, interp=rnd.INTERP_POLYM1)
x0, y0, f0 = rnd.getrnd(K, V, S=10, rf=0.05, t=0.25, interp=rnd.INTERP_POLYM3, densityrange=rnd.DENSITY_RANGE_EXTENDD)
# rnd.getrnd(K, V, S=10, rf=0.0, t=0.25, densityrange=rnd.DENSITY_RANGE_EXTENDD, interp=rnd.INTERP_POLYM9)

x1, y1, f1 = rnd.getrnd(K, V, S=10, rf=0.05, t=0.25, method=rnd.METHOD_STDR_EXTRADEN, interp=rnd.INTERP_POLYM3, extrap=rnd.EXTRAP_GPARTO, densityrange=rnd.DENSITY_RANGE_EXTENDD)
# x1, y1, f1 = rnd.getrnd(K, V, S=10, rf=0.05, t=0.25, method=rnd.METHOD_STDR_EXTRADEN, interp=rnd.INTERP_POLYM3, extrap=rnd.EXTRAP_GBETA2, densityrange=rnd.DENSITY_RANGE_EXTENDD)
x2, y2, f2 = rnd.getrnd(K, V, S=10, rf=0.05, t=0.25, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR1, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_EXTENDD)

plt.plot(x0,f0)#;plt.show()
plt.plot(x1,f1)#;plt.show()
plt.plot(x2,f2)#;plt.show()
plt.show()

rnd.getrnd(K, V, S=10, rf=0.0, t=0.25, densityrange=rnd.DENSITY_RANGE_DEFAULT)
rnd.getrnd(K, V, S=10, rf=0.0, t=0.25, densityrange=[10, 50])


# Can get a random sample, and test the the statistical difference between kurtotis
# VRP mostly tail risk in new literature.

