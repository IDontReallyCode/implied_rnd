"""
    ..\IV_SURFACE_FACTORS_LASSO\cleaned_bidaskiv
    
    contains a list of tickers that passed the cleaning.
    I can randomly select nticks=100 among the list, then select ndates=100 among the dates for each.
    
    I will then apply the three methods to get the RND.
    For each, I will calculate the 4 first central moments.
    For each fo the 4 central moments, I can calculate the three differences: extrapiv - extraden, extrapiv - extrahyp, extraden - extrahyp
    I can then statistically test the differences.
"""


import os
import random
import pickle
import numpy as np
import implied_rnd as rnd
import matplotlib.pyplot as plt

def main():
    # Set the seed for the random number generator
    random.seed(123)

    # Specify the directory
    dir_path = "..\\IV_SURFACE_FACTORS_LASSO\\cleaned_bidaskiv"

    # Get a list of all files in the directory
    all_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    # Randomly select nticks files
    nticks = 5
    ndates = 3
    matuty = 30
    selected_files = random.sample(all_files, nticks)
    # print(selected_files)

    for filename in selected_files:
        # Construct the full file path
        file_path = os.path.join(dir_path, filename)
        
        # Load the content of the file
        with open(file_path, 'rb') as f:
            content = pickle.load(f)
        
        # Get all keys from the dictionary
        all_keys = list(content.keys())
        
        # Randomly select ndates keys
        selected_keys = random.sample(all_keys, ndates)
        
        # print(f"Selected keys for file {filename}: {selected_keys}")    
        for thiskey in selected_keys:
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

            x0, y0, f0 = rnd.getrnd(K, V, S=S, rf=rf, t=t, interp=rnd.INTERP_POLYM3, densityrange=rnd.DENSITY_RANGE_EXTENDD)
            x1, y1, f1 = rnd.getrnd(K, V, S=S, rf=rf, t=t, method=rnd.METHOD_STDR_EXTRADEN, interp=rnd.INTERP_POLYM3, extrap=rnd.EXTRAP_GPARTO, densityrange=rnd.DENSITY_RANGE_EXTENDD)
            x2, y2, f2 = rnd.getrnd(K, V, S=S, rf=rf, t=t, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR1, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_EXTENDD)

            plt.plot(x0,f0)#;plt.show()
            plt.plot(x1,f1)#;plt.show()
            plt.plot(x2,f2)#;plt.show()
            plt.show()
            
            pass

    
    
    
#### __name__ MAIN()
if __name__ == '__main__':
    # SET = int(sys.argv[1])
    main()

    

