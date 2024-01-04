"""
    The objective here is to test the robustness of each method, using a regular stock that has lots of strikes and a wide range of log-moneyness.
    
    For one method:
    - We use the method to get the RND and then use numerical integration to get the first 4 moments, and consider those as the true values.
    - We test robustness on the left tail first
        - We remove one strike, apply the method, get the first 4 moments. Calculate the % change.
        - Redo removing one more strike.
        - repeat until removed half of the left strikes.
        Repeat for right tail
        Repeat removing on both sides
"""

import os
import random
import pickle
import numpy as np
import implied_rnd as rnd
import matplotlib.pyplot as plt
from scipy import stats


def main():
    dir_path = "..\\IV_SURFACE_FACTORS_LASSO\\cleaned_bidaskiv"
    filename = "AMZN.pkl"
    file_path = os.path.join(dir_path, filename)
    # Load the content of the file
    with open(file_path, 'rb') as f:
        content = pickle.load(f)
        
    # Get all keys from the dictionary
    all_keys = list(content.keys())
    totaldays = len(all_keys)
    inventorynumberofk = np.zeros((totaldays,2))
    inventorymoneyness = np.zeros((totaldays,2))
    
    matuty = 30
    
    for i, thiskey in enumerate(all_keys):
        # Get the numpy array
        array = np.array(content[thiskey]['dte'])
        
        # Find the index of the element in the array that is closest to the value
        index = (np.abs(array - matuty)).argmin()
        
        # Create a mask for the elements in the array that are closest to the value
        mask = (array == array[index])
        
        # print(f"Mask for key {thiskey}: {mask}")
        K = content[thiskey]['strikes'][mask]
        V = (content[thiskey]['ask_iv'][mask] + content[thiskey]['bid_iv'][mask])/2
        S = content[thiskey]['spot']
        rf = content[thiskey]['rf']
        t = content[thiskey]['dte'][0]/365
        
        m = np.multiply(1/np.sqrt(t), np.log(S*np.exp(rf*t)/K))
        
        number_check = K.shape[0]
        
        leftmask = K<S
        righmask = K>S
        
        nbleft = int(sum(leftmask))
        nbrigh = int(sum(righmask))
        
        inventorynumberofk[i,0] = nbleft
        inventorynumberofk[i,1] = nbrigh
        # if nbleft<=5:
        #     plt.plot(K,V)
        #     plt.show()
        #     pass
        inventorymoneyness[i,0] = np.min(m)
        inventorymoneyness[i,1] = np.max(m)
        
            
    print(f"We need {np.max(inventorynumberofk[:,0])} for the left")        
    print(f"We need {np.max(inventorynumberofk[:,1])} for the right")        
    print(f"We get {np.min(inventorynumberofk[:,0])} for the left")        
    print(f"We get {np.min(inventorynumberofk[:,1])} for the right")    
    
    print(f" moneyness, on average goes from {np.mean(inventorymoneyness[:,0])} to {np.mean(inventorymoneyness[:,1])}")
    # plt.hist(inventorynumberofk[:,0])
    # plt.show()
    # plt.hist(inventorynumberofk[:,1])
    # plt.show()
        
    pass
   
    
#### __name__ MAIN()
if __name__ == '__main__':
    # SET = int(sys.argv[1])
    main()

    

