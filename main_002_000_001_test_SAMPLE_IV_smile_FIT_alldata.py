import os
# import random
import polars as pl
import random
import matplotlib.pyplot as plt
import implied_rnd as rnd
import numpy as np

"""
2024-01-24:
In this program, I want to open a tick file and get the RND using different methods.

This can be used as an example for more advanced programs later.

As of 2024-01-24, rnd.py only deals with an Implied Volatility Curve (IVC) and not a surface (IVS).
Thus, for now, the program will use the closest maturity to 1-months horizon (30 days).
    => This is based on: Gagnon, Marie-Hélène / Power, Gabriel J.  Testing for changes in option-implied risk aversion, 2016-06 
    

"""

def main():
    # Set the seed for random number generation
    SEED = 357951

    directory = "E:/CBOE/ipc_per_tick"
    N = 10  # Number of dates to select

    # selected_file = 'SPX.ipc'
    ticker = 'SPX'
    selected_file = f'{ticker}.ipc'
    # selected_file = 'AMZN.ipc'
    # selected_file = 'ROKU.ipc'
    model1 = [rnd.INTERP_SVI000, "filtered w=0"]
    # model1 = [rnd.INTERP_SVI000, "SVI000"]
    model2 = [rnd.INTERP_SVI000, "all w=0"]
    model3 = [rnd.INTERP_SVI000, "all w=oi"]
    model4 = [rnd.INTERP_SVI000, "all w=iv"]
    # model2 = [rnd.INTERP_FACTR1, "FACTR1"]

    weighted = False

    file_path = os.path.join(directory, selected_file)
    df = pl.read_ipc(file_path)
    # print(df.columns)  # Print the columns of the DataFrame
    
    # Process the DataFrame as needed
    # Convert the 'quote_datetime' column to date only
    df = df.with_columns(pl.col('quote_datetime').cast(pl.Date).alias('quote_date'))
    
    # Pick N random dates from the quote_datetime column
    random_dates = df['quote_date'].unique().sample(N, seed=SEED).to_list()

    # import datetime
    # random_dates = [datetime.date(2019, 12, 31)]

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
        this_pf = this_pf.filter((pl.col('otm') == True)
                                 & (pl.col('implied_volatility') > 0)
                                 )                                                  ## CHECKED                                      

        filtered_pf = this_pf.filter((pl.col('dte') > 6) & (pl.col('bid') > 0.00) 
                                 & ((pl.col('ask') + pl.col('bid')) / 2 > 0.375)
                                 & (pl.col('otm') == True)
                                 & (pl.col('likely_stale') == False)
                                 & (pl.col('implied_volatility') > 0)
                                 )                                                  ## CHECKED                                      

        x = filtered_pf['tslm'].to_numpy()
        y = filtered_pf['implied_volatility'].to_numpy()
        w_oi = filtered_pf['open_interest'].to_numpy()
        w_oi = w_oi / w_oi.sum()

        w_iv = filtered_pf['implied_volatility'].to_numpy()
        w_iv = w_iv / w_iv.sum()

        xba = this_pf['tslm'].to_numpy()
        yb = this_pf['bid_iv'].to_numpy()
        ya = this_pf['ask_iv'].to_numpy()
        # stack the yb and ya vectors into yba
        yba = np.hstack((yb, ya))
        # stack xba twice to match the shape of yba and put that into xba
        xba = np.hstack((xba, xba))
        # now, remove any NaN values from xba and yba
        mask = np.isnan(yba)
        xba = xba[~mask]
        yba = yba[~mask]
        

        v_hat_1 = rnd.getfit(xba,yba, model1[0])
        v_hat_2 = rnd.getfit(x,y, model2[0])
        v_hat_3 = rnd.getfit(x,y, model3[0], weights=w_oi)
        v_hat_4 = rnd.getfit(x,y, model4[0], weights=w_iv)
        # plt.scatter(this_pf['tslm'], V_hat_poly3, s=10)
        # not do a scatter of the 'iv' vs 'tslm' columns
        plt.scatter(xba, yba, s=10, label = 'DATA')
        plt.scatter(xba, v_hat_1, s=10, label = model1[1])
        plt.scatter(x, v_hat_2, s=10, label = model2[1])
        plt.scatter(x, v_hat_3, s=10, label = model3[1])
        plt.scatter(x, v_hat_4, s=10, label = model4[1])
        # plt.scatter(xfiltered, yfiltered, s=10, label = 'DATA filtered')
        plt.legend()
        plt.title(f'{ticker} - {one_date}')
        plt.show()

        # # Save the variables x and y into two files named test_x.csv and test_y.csv
        # np.savetxt('test_x.csv', x, delimiter=',')
        # np.savetxt('test_y.csv', y, delimiter=',')


        pass


if __name__ == "__main__":
    main()
