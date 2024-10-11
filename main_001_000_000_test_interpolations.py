import implied_rnd as rnd
import numpy as np
import matplotlib.pyplot as plt

K = np.array([3,4,5,6,7,8,9,10,11,12,13,14,15])
V = np.array([.7256,.724,.723,.722,.721,.720,.720,.720,.7202, .7204, .7206, .7208, .721])

V_hat = rnd.getfit(K, V, rnd.INTERP_POLYM3)

# now, plot the original and the interpolated
plt.plot(K,V)
plt.plot(K,V_hat)
plt.show()

pause = 1