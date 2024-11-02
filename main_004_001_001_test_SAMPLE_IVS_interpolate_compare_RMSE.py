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
    iv = data['iv']

    x = data['x']

    model = data['model']

    # Fit the model to the data
    v_hat = rnd.getfit(x, iv, model)

    if np.any(np.isnan(v_hat)):
        pause=1

    rmse = np.sqrt( np.sum((v_hat-iv)*(v_hat-iv) / len(iv) ))

    return rmse



def main():
    # Set the seed for random number generation
    random.seed(35718)

    directory = "E:/CBOE/ipc_per_tick"
    N = 1000  # Number of dates to select

    # selected_file = 'SPX.ipc'
    ticker = 'QQQ'
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
    random_dates = df['quote_date'].unique().sample(N, seed=357951).to_list()

    # import datetime
    # random_dates = [datetime.date(2019, 12, 31)]
    
    # We will collect N results for each of the 4 models
    results = {}
    models = [rnd.INTERP_3D_M2VOL, rnd.INTERP_3D_M2VAR, rnd.INTERP_3D_FGVGG, rnd.INTERP_3D_SVI01]

    # for model in ['M2vol', 'M2var', 'FGVGG', 'SVI01']:
    #     results[model] = []

    # Initialize a dictionary to store the data for parallel processing
    M2VOL_list = []
    M2VAR_list = []
    FGVGG_list = []
    SVI01_list = []

    for one_date in random_dates:

        # Filter the DataFrame to only include rows where the quote_datetime column is equal to the random_date selected        ## CHECKED
        this_pf = df.filter(pl.col('quote_date') == one_date)
        
        # filter only the rows for which 'quote_datetime' is 11:30                                                              ## CHECKED
        this_pf = this_pf.filter(pl.col('quote_datetime').str.contains('11:30'))        
        
        # Remove the 'dte_diff' (and other extra columns)                                                                       ## CHECKED      
        # this_pf = this_pf.drop(['dte_diff', 'root', 'open', 'high', 'low', 'close', 'bid_size', 'ask_size', 'delta', 'gamma', 'theta', 'vega', 'rho'], ignore_missing=True)
        # this_pf = this_pf.drop(['dte_diff', 'root', 'open', 'high', 'low', 'close', 'bid_size', 'ask_size', 'delta', 'gamma', 'theta', 'vega', 'rho', 'eod', 'in1yearrange', 'quote_date'])
            
        """
        Options with the following characteristics are removed: 
        (i) a time‐to‐maturity shorter than 6 days, 
        (ii) a price lower than $3/8, 
        (iii) a zero bid price,
        and (iv) options with a bid–ask spread larger than 175% of the option's midprice
        """
        this_pf = this_pf.filter((pl.col('dte') > 14) & (pl.col('bid') > 0.00) 
                                 & ((pl.col('ask') + pl.col('bid')) / 2 > 0.375)
                                 & (pl.col('otm') == True)
                                 & (pl.col('likely_stale') == False)
                                 & (pl.col('implied_volatility') > 0.05)
                                 )                                                  ## CHECKED                                      
        # & (pl.col('trade_volume') > 0)
        # & (pl.col('otm') == True)
        # & ((pl.col('ask') - pl.col('bid')) / ((pl.col('ask') + pl.col('bid')) / 2) < 1.75) 
        

        # Now, I want a surface plot of the IVS with tslm, dte/365, and implied_volatility

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        
        if len(this_pf) < 20:
            continue

        # Create a meshgrid for tslm and dte/365
        tslm = this_pf['tslm'].to_numpy()
        dte = this_pf['dte'].to_numpy() / 365
        iv = this_pf['implied_volatility'].to_numpy()

        # # ax.scatter(tslm, dte, iv, c=iv, cmap='viridis', marker='o')
        # ax.scatter(tslm, dte, iv, marker='.', label='Original Data')

        # ax.set_xlabel('TSLM')
        # ax.set_ylabel('DTE/365')
        # ax.set_zlabel('Implied Volatility')
        # ax.set_title(f'IV Scatter Plot for {ticker} on {one_date}')

        # Now, I want to fit a surface to the IVS
        x = np.column_stack((tslm, dte))

        # collect the data in the dict
        M2VOL_list.append({'x': x, 'iv': iv, 'model': rnd.INTERP_3D_M2VOL})
        M2VAR_list.append({'x': x, 'iv': iv, 'model': rnd.INTERP_3D_M2VAR})
        FGVGG_list.append({'x': x, 'iv': iv, 'model': rnd.INTERP_3D_FGVGG})
        SVI01_list.append({'x': x, 'iv': iv, 'model': rnd.INTERP_3D_SVI01})

        

        # # xout = np.column_stack((np.linspace(tslm.min(), tslm.max(), 100), np.full(100, 30/365)))
        # # v_hat_3D_poly2 = rnd.getfit(x, iv, rnd.INTERP_3D_M2VOL, xout)
        # v_hat_3D_M2vol = rnd.getfit(x, iv, rnd.INTERP_3D_M2VOL)
        # v_hat_3D_M2var = rnd.getfit(x, iv, rnd.INTERP_3D_M2VAR)
        # v_hat_3D_FGVGG = rnd.getfit(x, iv, rnd.INTERP_3D_FGVGG)
        # # v_hat_3D_SVI00 = rnd.getfit(x, iv, rnd.INTERP_3D_SVI00)
        # v_hat_3D_SVI01 = rnd.getfit(x, iv, rnd.INTERP_3D_SVI01)

        # # scatter plot v_hat
        # # ax.scatter(xout[:,0], xout[:,1], v_hat_3D_poly2, marker='o', label='Polynomial Fit')
        # # ax.scatter(x[:,0], x[:,1], v_hat_3D_M2vol, marker='o', label='Polynomial Fit on vol')
        # # ax.scatter(x[:,0], x[:,1], v_hat_3D_M2var, marker='o', label='Polynomial Fit on var')
        # # ax.scatter(x[:,0], x[:,1], v_hat_3D_FGVGG, marker='o', label='FGVGG')
        # # ax.scatter(x[:,0], x[:,1], v_hat_3D_SVI00, marker='o', label='SVI03D')
        # # ax.scatter(x[:,0], x[:,1], v_hat_3D_SVI01, marker='o', label='SVI03D')
        # # ax.legend()
        # # plt.show()

        # # Append the results to the results dictionary
        # results['M2vol'].append(np.sqrt( np.sum((v_hat_3D_M2vol-iv)*(v_hat_3D_M2vol-iv) / len(iv) )))
        # results['M2var'].append(np.sqrt( np.sum((v_hat_3D_M2var-iv)*(v_hat_3D_M2var-iv) / len(iv) )))
        # results['FGVGG'].append(np.sqrt( np.sum((v_hat_3D_FGVGG-iv)*(v_hat_3D_FGVGG-iv) / len(iv) )))
        # # results['SVI00'].append(np.sqrt( np.sum((v_hat_3D_SVI00-iv)*(v_hat_3D_SVI00-iv) / len(iv) )))
        # results['SVI01'].append(np.sqrt( np.sum((v_hat_3D_SVI01-iv)*(v_hat_3D_SVI01-iv) / len(iv) )))

        # pass

    # fitsmanual = []
    # for day in M2VAR_list:
    #     fitsmanual.append(getRMSE(day))

    # start a pool for prarallel processing
    with mp.Pool(10) as p:
        # Run the parallel processing
        fits_M2VOL = p.map(getRMSE, M2VOL_list)
        fits_M2VAR = p.map(getRMSE, M2VAR_list)
        fits_FGVGG = p.map(getRMSE, FGVGG_list)
        fits_SVI01 = p.map(getRMSE, SVI01_list)

    # fits_SVI01 = []
    # for day in SVI01_list:
    #     fits_SVI01.append(getRMSE(day))                                            

    # regroupd the results into a results dictionary
    results['M2vol'] = fits_M2VOL
    results['M2var'] = fits_M2VAR
    results['FGVGG'] = fits_FGVGG
    results['SVI01'] = fits_SVI01

    # Print the results
    print("Root Mean Square Error (RMSE) for Different Models:")
    print("---------------------------------------------------")
    for model in results.keys():
        mean_rmse = np.mean(results[model])
        std_rmse = np.std(results[model])
        min_rmse = np.min(results[model])
        max_rmse = np.max(results[model])
        print(f'{model:6} | Mean: {mean_rmse:10.6f} | Std: {std_rmse:10.6f} | Min: {min_rmse:10.6f} | Max: {max_rmse:10.6f}')
        # print(f'{model}: {np.mean(results[model])}')



if __name__ == "__main__":
    main()
