import implied_rnd
import numpy as np

K = np.array([3,4,5,6,7,8,9,10,11,12,13,14,15])
V = np.array([25,24,23,22,21,20,20,20,20.2, 20.4, 20.6, 20.8, 21])

implied_rnd.getrnd(K, V)

K = np.array([3,4,5,6,7,8,9,10,11,12,13,14,15])
V = np.array([25,24,23,22,21,20,20,20,20.2, 20.4, 20.6, 20.8, 21, 34])

try:
    implied_rnd.getrnd(K, V)
except ValueError as e:
    print(f"it failed: {e}")


K = [3,4,5,6,7,8,9,10,11,12,13,14,15]
V = [25,24,23,22,21,20,20,20,20.2, 20.4, 20.6, 20.8, 21]

try:
    implied_rnd.getrnd(K, V)
    print("using lists worked as well")
except ValueError as e:
    print(f"it failed: {e}")




