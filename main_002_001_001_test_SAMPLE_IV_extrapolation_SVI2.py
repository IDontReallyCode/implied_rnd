import os
import random
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
    # random.seed(357951)
    SEED = 357951

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
    random_dates = df['quote_date'].unique().sample(N, seed=SEED).to_list()

    # import datetime
    # random_dates = [datetime.date(2019, 12, 31)]

    # I want to remove the first entry in random_dates
    # random_dates = random_dates[1:]

    for index, one_date in enumerate(random_dates):

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
        # & (pl.col('trade_volume') > 0)
        # & (pl.col('otm') == True)
        # & ((pl.col('ask') - pl.col('bid')) / ((pl.col('ask') + pl.col('bid')) / 2) < 1.75) 

        x = this_pf['tslm'].to_numpy()
        y = this_pf['implied_volatility'].to_numpy()

        w_oi = this_pf['open_interest'].to_numpy()
        w_oi = w_oi / w_oi.sum()

        w_iv = -1*this_pf['implied_volatility'].to_numpy()
        w_iv = w_iv - w_iv.min()
        w_iv = w_iv / w_iv.sum()

        # calculate the normal density for a standard normal distribution
        w_nm = np.exp(-0.5*x**2)

        # not do a scatter of the 'iv' vs 'tslm' columns
        # plt.scatter(this_pf['tslm'], this_pf['bid_iv'])
        # plt.scatter(this_pf['strike'], this_pf['implied_volatility'])

        # V_hat_poly3 = rnd.getfit(this_pf['tslm'].to_numpy(), this_pf['implied_volatility'].to_numpy(), rnd.INTERP_POLYM3)
        # V_hat_nlin0 = rnd.getfit(this_pf['tslm'].to_numpy(), this_pf['implied_volatility'].to_numpy(), rnd.INTERP_SVI100)
        # V_hat_nlin1 = rnd.getfit(this_pf['tslm'].to_numpy(), this_pf['implied_volatility'].to_numpy(), rnd.INTERP_SVI102)

        # Now, I want to extend the domain in tslm by 25% on each side and use N points
        N = 300
        tslm_min = -5 #this_pf['tslm'].min()*20
        tslm_max = +2 #this_pf['tslm'].max()*2.5
        tslm_range = tslm_max - tslm_min
        tslm_new = np.linspace(tslm_min, tslm_max, N)

        V_hat_extr0 = rnd.getfitextrapolated(x,y, tslm_new, rnd.INTERP_POLYM4)
        V_hat_extr1 = rnd.getfitextrapolated(x,y, tslm_new, rnd.INTERP_SVI000)
        V_hat_extr2 = rnd.getfitextrapolated(x,y, tslm_new, rnd.INTERP_POLYM4, weights=w_nm)


        # plt.scatter(this_pf['tslm'], V_hat_poly3, s=10)
        # plt.scatter(this_pf['tslm'], V_hat_nlin0, s=10)
        # plt.scatter(this_pf['tslm'], V_hat_nlin1, s=10)
        plt.scatter(this_pf['tslm'], this_pf['implied_volatility'], s=10)
        plt.plot(tslm_new, V_hat_extr0, alpha=0.5)
        plt.plot(tslm_new, V_hat_extr1, alpha=0.5)
        plt.plot(tslm_new, V_hat_extr2, alpha=0.5)
        plt.legend(['data', 'INTERP_POLYM4', 'INTERP_SVI000', 'INTERP_POLYM4 w'])
        plt.title(f'#{index}:  {ticker} - {one_date}')
        plt.ylim(0, 1)
        plt.show()

        # # I want to put this_pf['tslm'], this_pf['implied_volatility']**2 into numpy values called x and y
        # x = this_pf['tslm'].to_numpy()
        # y = this_pf['implied_volatility'].to_numpy()**2
        # # Now I want to save x and y into a file I can easily load up into numpy later
        # np.save(f'{ticker}_{one_date}_x.npy', x)
        # np.save(f'{ticker}_{one_date}_y.npy', y)
        # now if I want to load them up later, I can do:
        # x = np.load(f'{ticker}_{one_date}_x.npy')
        # y = np.load(f'{ticker}_{one_date}_y.npy')

        pass


if __name__ == "__main__":
    main()
