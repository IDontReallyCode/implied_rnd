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
from scipy import stats

def main():
    # Set the seed for the random number generator
    random.seed(123)

    # Specify the directory
    dir_path = "..\\IV_SURFACE_FACTORS_LASSO\\cleaned_bidaskiv"

    # Get a list of all files in the directory
    all_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    # Randomly select nticks files
    nticks = 100
    ndates = 100
    matuty = 30
    selected_files = random.sample(all_files, nticks)
    # print(selected_files)
    
    metrics0 = np.zeros((nticks*ndates, 4))
    metrics1 = np.zeros((nticks*ndates, 4))
    metrics2 = np.zeros((nticks*ndates, 4))
    
    diff01 = np.zeros((nticks*ndates, 4))
    diff02 = np.zeros((nticks*ndates, 4))
    diff12 = np.zeros((nticks*ndates, 4))
    
    i = 0

    for filename in selected_files:
        # Construct the full file path
        file_path = os.path.join(dir_path, filename)
        
        # Load the content of the file
        with open(file_path, 'rb') as f:
            content = pickle.load(f)
        
        # Get all keys from the dictionary
        all_keys = list(content.keys())
        
        samplesize = int(min(ndates,len(all_keys)/2))
        
        # Randomly select ndates keys
        selected_keys = random.sample(all_keys, samplesize)
        
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
            
            number_check = K.shape[0]
            
            if number_check>5:
                x0, y0, f0 = rnd.getrnd(K, V, S=S, rf=rf, t=t, interp=rnd.INTERP_POLYM3, densityrange=rnd.DENSITY_RANGE_EXTENDD)
                x1, y1, f1 = rnd.getrnd(K, V, S=S, rf=rf, t=t, method=rnd.METHOD_STDR_EXTRADEN, interp=rnd.INTERP_POLYM3, extrap=rnd.EXTRAP_GPARTO, densityrange=rnd.DENSITY_RANGE_EXTENDD)
                x2, y2, f2 = rnd.getrnd(K, V, S=S, rf=rf, t=t, method=rnd.METHOD_TLSM_EXTRAPIV, interp=rnd.INTERP_FACTR1, extrap=rnd.EXTRAP_ASYMPT, densityrange=rnd.DENSITY_RANGE_EXTENDD)

                # plt.plot(x0,f0)#;plt.show()
                # plt.plot(x1,f1)#;plt.show()
                # plt.plot(x2,f2)#;plt.show()
                # plt.show()
                
                # cm01, cm02, cm03, cm04 = rnd.getmoments(x0,f0)
                # cm11, cm12, cm13, cm14 = rnd.getmoments(x1,f1)
                # cm21, cm22, cm23, cm24 = rnd.getmoments(x2,f2)
                
                metrics0[i,:] = np.array([rnd.getmoments(x0,f0)])
                metrics1[i,:] = np.array([rnd.getmoments(x1,f1)])
                metrics2[i,:] = np.array([rnd.getmoments(x2,f2)])
                # print([cm01, cm02, cm03, cm04])
                # print([cm11, cm12, cm13, cm14])
                # print([cm21, cm22, cm23, cm24])
                diff01[i,:] = (metrics1[i,:] - metrics0[i,:])/metrics0[i,:]
                diff02[i,:] = (metrics2[i,:] - metrics0[i,:])/metrics0[i,:]
                diff12[i,:] = (metrics2[i,:] - metrics1[i,:])/metrics1[i,:]
            
                i+=1
            pass
    
    diff01 = diff01[:i,:]
    diff02 = diff02[:i,:]
    diff12 = diff12[:i,:]
        
    # Now, we want to calculate the statistics
    # Assuming diff01 is your numpy array
    t_stats, p_values = stats.ttest_1samp(diff01, 0)
    print(f"dumb VS Gen-Pareto T-statistics: {t_stats}")
    print(f"dumb VS Gen-Pareto p-values: {p_values}")
    t_stats, p_values = stats.ttest_1samp(diff02, 0)
    print(f"dumb VS TSLM T-statistics: {t_stats}")
    print(f"dumb VS TSLM p-values: {p_values}")
    t_stats, p_values = stats.ttest_1samp(diff12, 0)
    print(f"Gen-Pareto VS TSLM T-statistics: {t_stats}")
    print(f"Gen-Pareto VS TSLM p-values: {p_values}")
    print("")
    
    
    
    
    
    
#### __name__ MAIN()
if __name__ == '__main__':
    # SET = int(sys.argv[1])
    main()

    

