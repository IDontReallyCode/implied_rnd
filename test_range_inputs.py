import implied_rnd as rnd
import numpy as np

K = np.array([3,4,5,6,7,8,9,10,11,12,13,14,15])
V = np.array([25,24,23,22,21,20,20,20,20.2, 20.4, 20.6, 20.8, 21])

rnd.getrnd(K, V, densityrange=rnd.DENSITY_RANGE_EXTENDD, S=10, rf=0.0)
rnd.getrnd(K, V, densityrange=rnd.DENSITY_RANGE_DEFAULT, S=10, rf=0.0)
rnd.getrnd(K, V, densityrange=[10, 50], S=10, rf=0.0)



