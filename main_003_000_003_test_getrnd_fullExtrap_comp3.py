import os
import random
import polars as pl
import random
import matplotlib.pyplot as plt
import implied_rnd as rnd


"""
2024-01-24:
In this program, I want to open a ticker file and get the RND using different methods.

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
        this_pf = this_pf.filter((pl.col('dte') > 6) & (pl.col('bid') > 0.00) 
                                 & ((pl.col('ask') + pl.col('bid')) / 2 > 0.375)
                                 & (pl.col('otm') == True)
                                 & (pl.col('likely_stale') == False)
                                 & (pl.col('implied_volatility') > 0)
                                 )                                                  ## CHECKED                                      
        # & (pl.col('trade_volume') > 0)
        # & (pl.col('otm') == True)
        # & ((pl.col('ask') - pl.col('bid')) / ((pl.col('ask') + pl.col('bid')) / 2) < 1.75) 
        K = this_pf['strike'].to_numpy()
        V = this_pf['implied_volatility'].to_numpy()
        S = float(this_pf['active_underlying_price'].to_numpy()[0])
        rf = float(this_pf['rf'].to_numpy()[0])
        t = float(this_pf['dte'].to_numpy()[0] / 365)
        w = this_pf['open_interest'].to_numpy()
        w = w / w.sum()


        # x0, y0, f0 = rnd.getrnd(K, V, S=S, rf=rf, t=t, interp=rnd.INTERP_POLYM4, densityrange=rnd.DENSITY_RANGE_DEFAULT, method=rnd.METHOD_STDR_EXTRADEN, extrap=rnd.EXTRAP_GP3PTS, fittingweights=w)
        x1, y1, f1_scaled, f1 = rnd.getrnd(K, V, S=S, rf=rf, t=t, interp=rnd.INTERP_POLYM4, densityrange=rnd.DENSITY_RANGE_DEFAULT, method=rnd.METHOD_STDR_EXTRADEN, extrap=rnd.EXTRAP_GEV3PT, fittingweights=w)
        x2, y2, f2_scaled, f2 = rnd.getrnd(K, V, S=S, rf=rf, t=t, interp=rnd.INTERP_SVI001, densityrange=rnd.DENSITY_RANGE_DEFAULT, method=rnd.METHOD_TLSM_EXTRAPIV, extrap=rnd.EXTRAP_ASYMPT)

        # plt.plot(x0, f0, label="M4 G pareto")
        plt.plot(x1, f1_scaled, label="M4 GEV")
        plt.plot(x2, f2_scaled, label="SVI")
        plt.legend()
        plt.title(f'{ticker} - {one_date}')
        plt.show()
        pass


if __name__ == "__main__":
    main()
