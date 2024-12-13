import os
import random
import polars as pl
import random
import matplotlib.pyplot as plt
import implied_rnd as rnd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp


"""
2024-10-23:
In this program, I want to open a tick file and get the RND using different methods.

This can be used as an example for more advanced programs later.

I will use this to work on rnd.py and the interpolation of the IVS.
    

"""


def getRMSE(data)->dict:
    iv = data['V']

    x = data['M']

    # w = data['w']

    model = data['model']

    # Fit the model to the data
    v_hat = rnd.getfit(x, iv, model)

    if np.any(np.isnan(v_hat)):
        pause=1

    rmse = np.sqrt( np.sum((v_hat-iv)*(v_hat-iv) / len(iv) ))

    return rmse



def main():
    # Set the seed for random number generation
    SEED=35718

    directory = "E:/CBOE/ipc_per_tick"
    N = 2000  # Number of dates to select

    # selected_file = 'SPX.ipc'
    ticker = 'SPX'
    selected_file = f'{ticker}.ipc'
    # selected_file = 'AMZN.ipc'
    # selected_file = 'ROKU.ipc'

    file_path = os.path.join(directory, selected_file)
    df = pl.read_ipc(file_path)
    # print(df.columns)  # Print the columns of the DataFrame
    
    # Process the DataFrame as needed
    # Convert the 'quote_datetime' column to date only
    df = df.with_columns(pl.col('quote_datetime').cast(pl.Date).alias('quote_date'))
    df = df.drop(['dte_diff', 'root', 'open', 'high', 'low', 'close', 'bid_size', 'ask_size', 'delta', 'gamma', 'theta', 'vega', 'rho', 'eod', 'in1yearrange'])
        
    # Pick N random dates from the quote_datetime column
    random_dates = df['quote_date'].unique().sample(N, seed=SEED).to_list()

    # import datetime
    # random_dates = [datetime.date(2019, 12, 31)]
    
    # We will collect N results for each of the models
    # modelstocompare = [rnd.INTERP_POLYM4, rnd.INTERP_SVI000]
    modelstocompare = [rnd.INTERP_POLYM4, rnd.INTERP_SVI000, rnd.INTERP_SVI001]
    results = {}
    workingdata = {}

    # loop on models and adda key to the results dictionary
    for model in modelstocompare:
        results[model] = []
        workingdata[model] = []


    for one_date in random_dates:

        # Filter the DataFrame to only include rows where the quote_datetime column is equal to the random_date selected        ## CHECKED
        this_pf = df.filter(pl.col('quote_date') == one_date)
        
        # filter only the rows for which 'quote_datetime' is 11:30                                                              ## CHECKED
        this_pf = this_pf.filter(pl.col('quote_datetime').str.contains('11:30'))        
        
        # now keep only the rows for which 'dte' is closest to 30 days                                                          ## CHECKED
        # Calculate the absolute difference between 'dte' and 30
        this_pf = this_pf.with_columns((pl.col('dte') - 30).abs().alias('dte_diff'))
        # Keep only the rows where 'dte' is closest to 30
        this_pf = this_pf.filter(pl.col('dte_diff') == this_pf['dte_diff'].min())

        # Remove the 'dte_diff' (and other extra columns)                                                                       ## CHECKED      
        # this_pf = this_pf.drop(['dte_diff', 'root', 'open', 'high', 'low', 'close', 'bid_size', 'ask_size', 'delta', 'gamma', 'theta', 'vega', 'rho'], ignore_missing=True)
        this_pf = this_pf.drop(['dte_diff', 'root', 'open', 'high', 'low', 'close', 'bid_size', 'ask_size', 'delta', 'gamma', 'theta', 'vega', 'rho', 'eod', 'in1yearrange', 'quote_date'])
            
        """
        Options with the following characteristics are removed: 
        (i) a time‐to‐maturity shorter than 6 days, 
        (ii) a price lower than $3/8, 
        (iii) a zero bid price,
        and (iv) options with a bid–ask spread larger than 175% of the option's midprice
        """
        this_pf = this_pf.filter((pl.col('dte') > 6) & (pl.col('bid') > 0.00) 
                                 & ((pl.col('ask') + pl.col('bid')) / 2 > 0.375)
                                 & (pl.col('otm') == True)
                                 & (pl.col('likely_stale') == False)
                                 & (pl.col('implied_volatility') > 0)
                                 )                                                  ## CHECKED                                      
        
        if len(this_pf) < 10:
            continue

        # Create a meshgrid for tslm and dte/365
        M = this_pf['tslm'].to_numpy()
        V = this_pf['implied_volatility'].to_numpy()
        # w = this_pf['open_interest'].to_numpy()
        # w = w / w.sum()

        # loop in the keys in results. for each of them, add a dictionary with the data and the modelID
        for model in modelstocompare:
            workingdata[model].append({'M': M, 'V': V, 'model': model})

        pass


    # Here we test getting one RMSE value for each model
    for model in modelstocompare:
        results[model].append(getRMSE(workingdata[model][0]))

    # start a pool for prarallel processing
    with mp.Pool(10) as p:
        # loop on the keys in results and get the RMSE for each of them
        for model in modelstocompare:
            results[model] = p.map(getRMSE, workingdata[model])

    # print the results side-by-side
    # I want the mean, std, min and max for each model, and it needs to be shown side by side
    # so for simpicity, we will collect results first, print later
    results_summary = {}
    for model in modelstocompare:
        results_summary[model] = {'mean': np.mean(results[model]), 'std': np.std(results[model]), 'min': np.min(results[model]), 'max': np.max(results[model])}

    # print the results
    print('Model'.ljust(20), 'Mean'.ljust(10), 'Std'.ljust(10), 'Min'.ljust(10), 'Max'.ljust(10))
    for model in modelstocompare:
        print(str(model).ljust(20), str(results_summary[model]['mean']).ljust(10), str(results_summary[model]['std']).ljust(10), str(results_summary[model]['min']).ljust(10), str(results_summary[model]['max']).ljust(10))

    pass


if __name__ == "__main__":
    main()


"""
SPX
2000 random dates
SEED = 35718
Model                Mean       Std        Min        Max       
14                   0.0041291458 0.0029646251 0.00050228264 0.033175297
2040                 0.0040636025 0.0031125052 0.0005473119 0.048256133
2041                 0.0037108127 0.0024675746 0.0005299807 0.029629953

"""