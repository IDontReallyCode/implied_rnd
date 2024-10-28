import os
import random
import polars as pl
import random
import matplotlib.pyplot as plt
import implied_rnd as rnd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

"""
2024-10-23:
In this program, I want to open a tick file and get the RND using different methods.

This can be used as an example for more advanced programs later.

I will use this to work on rnd.py and the interpolation of the IVS.
    

"""

def main():
    # Set the seed for random number generation
    random.seed(357951)

    directory = "E:/CBOE/ipc_per_tick"
    N = 10  # Number of dates to select

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
    
    # Pick N random dates from the quote_datetime column
    random_dates = df['quote_date'].unique().sample(N).to_list()

    # import datetime
    # random_dates = [datetime.date(2019, 12, 31)]

    for one_date in random_dates:

        # Filter the DataFrame to only include rows where the quote_datetime column is equal to the random_date selected        ## CHECKED
        this_pf = df.filter(pl.col('quote_date') == one_date)
        
        # filter only the rows for which 'quote_datetime' is 11:30                                                              ## CHECKED
        this_pf = this_pf.filter(pl.col('quote_datetime').str.contains('11:30'))        
        
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
        this_pf = this_pf.filter((pl.col('dte') > 14) & (pl.col('bid') > 0.00) 
                                 & ((pl.col('ask') + pl.col('bid')) / 2 > 0.375)
                                 & (pl.col('otm') == True)
                                 & (pl.col('likely_stale') == False)
                                 & (pl.col('implied_volatility') > 0)
                                 )                                                  ## CHECKED                                      
        # & (pl.col('trade_volume') > 0)
        # & (pl.col('otm') == True)
        # & ((pl.col('ask') - pl.col('bid')) / ((pl.col('ask') + pl.col('bid')) / 2) < 1.75) 
        

        # Now, I want a surface plot of the IVS with tslm, dte/365, and implied_volatility

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a meshgrid for tslm and dte/365
        tslm = this_pf['tslm'].to_numpy()
        dte = this_pf['dte'].to_numpy() / 365
        iv = this_pf['implied_volatility'].to_numpy()

        # ax.scatter(tslm, dte, iv, c=iv, cmap='viridis', marker='o')
        ax.scatter(tslm, dte, iv, marker='.', label='Original Data')

        ax.set_xlabel('TSLM')
        ax.set_ylabel('DTE/365')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(f'IV Scatter Plot for {ticker} on {one_date}')

        # Now, I want to fit a surface to the IVS
        x = np.column_stack((tslm, dte))
        xout = np.column_stack((np.linspace(tslm.min(), tslm.max(), 100), np.full(100, 30/365)))
        v_hat_3D_poly2 = rnd.getfit(x, iv, rnd.INTERP_3D_POLY2, xout)
        v_hat_3D_poly2 = rnd.getfit(x, iv, rnd.INTERP_3D_POLY2, xout)

        # scatter plot v_hat
        ax.scatter(xout[:,0], xout[:,1], v_hat_3D_poly2, marker='o', label='Polynomial Fit')
        ax.legend()


        plt.show()

        # not do a scatter of the 'iv' vs 'tslm' columns
        # plt.scatter(this_pf['tslm'], this_pf['bid_iv'])
        # plt.scatter(this_pf['strike'], this_pf['implied_volatility'])
        # plt.scatter(this_pf['tslm'], this_pf['implied_volatility'], s=10)

        # # V_hat_poly3 = rnd.getfit(this_pf['tslm'].to_numpy(), this_pf['implied_volatility'].to_numpy(), rnd.INTERP_POLYM3)
        # V_hat_nlin0 = rnd.getfit(this_pf['tslm'].to_numpy(), this_pf['implied_volatility'].to_numpy(), rnd.INTERP_SVI000)
        # V_hat_nlin1 = rnd.getfit(this_pf['tslm'].to_numpy(), this_pf['implied_volatility'].to_numpy(), rnd.INTERP_SVI002)

        # # Now, I want to extend the domain in tslm by 25% on each side and use N points
        # N = 100
        # tslm_min = this_pf['tslm'].min()*1.25
        # tslm_max = this_pf['tslm'].max()*1.25
        # tslm_range = tslm_max - tslm_min
        # tslm_new = np.linspace(tslm_min, tslm_max, N)

        # V_hat_extr0 = rnd.getfitextrapolated(this_pf['tslm'].to_numpy(), this_pf['implied_volatility'].to_numpy(), tslm_new, rnd.INTERP_SVI000)
        # V_hat_extr1 = rnd.getfitextrapolated(this_pf['tslm'].to_numpy(), this_pf['implied_volatility'].to_numpy(), tslm_new, rnd.INTERP_SVI002)


        # # plt.scatter(this_pf['tslm'], V_hat_poly3, s=10)
        # # plt.scatter(this_pf['tslm'], V_hat_nlin0, s=10)
        # # plt.scatter(this_pf['tslm'], V_hat_nlin1, s=10)
        # plt.plot(tslm_new, V_hat_extr0, alpha=0.5)
        # plt.plot(tslm_new, V_hat_extr1, alpha=0.5)
        # plt.legend(['data', 'INTERP_SVI000', 'INTERP_SVI002'])
        # plt.title(f'{ticker} - {one_date}')
        # plt.show()

        pass


if __name__ == "__main__":
    main()
