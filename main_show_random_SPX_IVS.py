import os
import random
import polars as pl
import random
import matplotlib.pyplot as plt

"""
The objective is simply to show visually a set of random Implied Volatility Surfaces (IVSs).

2024-01-02: This is a work in progress.  For now, this will simply show a truly random set of IVSs, selecting randomly the ticker, and selecting randomly the days.
2024-01-24: This shows specifically IVS from SPX. It randomly selects N days, and then for each day, it picks each time, and then it shows the IVS for that day and time.
"""

def main():
    # Set the seed for random number generation
    random.seed(84)

    directory = "E:/CBOE/ipc_per_tick"
    N = 5  # Number of dates to select

    selected_file = 'SPX.ipc'

    file_path = os.path.join(directory, selected_file)
    df = pl.read_ipc(file_path)
    # print(df.columns)  # Print the columns of the DataFrame
    
    # Process the DataFrame as needed
    # Convert the 'quote_datetime' column to date only
    df = df.with_columns(pl.col('quote_datetime').cast(pl.Date).alias('quote_date'))
    
    # Pick N random dates from the quote_datetime column
    random_dates = df['quote_date'].unique().sample(N).to_list()

    for one_date in random_dates:

        # Filter the DataFrame to only include rows where the quote_datetime column is equal to the random_date
        df_one_date = df.filter(pl.col('quote_date') == one_date)
        
        # now get all unique 'quote_datetime' values
        unique_datetimes = df_one_date['quote_datetime'].unique().to_list()
        
        # loop over all unique_dates
        for one_datetime in unique_datetimes:
            # filter the DataFrame to only include rows where the quote_datetime column is equal to the random value
            df_one_time_OTM = df_one_date.filter((pl.col('quote_datetime') == one_datetime)  & (pl.col('otm') == False)
                                & (pl.col('ask_iv') > 0.0))
            df_one_time_ITM = df_one_date.filter((pl.col('quote_datetime') == one_datetime)  & (pl.col('otm') == True)
                                & (pl.col('ask_iv') > 0.0))
            
            """
            Options with the following characteristics are removed: 
            (i) a time‐to‐maturity shorter than 6 days, 
            (ii) a price lower than $3/8, 
            (iii) a zero bid price,
            and (iv) options with a bid–ask spread larger than 175% of the option's midprice
            """
            # from df_one_time, keep only those for which the 'dte' is greater then 6, 'bid' is greater than 0, ('ask'-'bid')/('ask'+'bid')/2 < 1.75
            df_one_timefiltered = df_one_time_OTM.filter((pl.col('dte') > 6) & (pl.col('bid') > 0) & ((pl.col('ask') - pl.col('bid')) / ((pl.col('ask') + pl.col('bid')) / 2) < 1.75) & ((pl.col('ask') + pl.col('bid')) / 2 > 0.375))
            df_one_timefilteredITM = df_one_time_ITM.filter((pl.col('implied_volatility') >  0) & (pl.col('dte') > 6) & (pl.col('bid') > 0) & ((pl.col('ask') - pl.col('bid')) / ((pl.col('ask') + pl.col('bid')) / 2) < 1.75) & ((pl.col('ask') + pl.col('bid')) / 2 > 0.375))
            # now get the inverse of that filter
            df_one_timefilteredout = df_one_time_OTM.filter((pl.col('dte') > 6) & (pl.col('bid') > 0) & ((pl.col('ask') - pl.col('bid')) / ((pl.col('ask') + pl.col('bid')) / 2) >= 1.75))
            
            # Create a new plot with 'tslm' on x-axis and 'ask_iv' on y-axis
            plt.scatter(df_one_timefilteredITM['tslm'], df_one_timefilteredITM['implied_volatility'], color='green', label='ITM', s=10)
            plt.scatter(df_one_timefiltered['tslm'], df_one_timefiltered['implied_volatility'], color='blue', label='Filtered', s=10)
            plt.scatter(df_one_timefilteredout['tslm'], df_one_timefilteredout['implied_volatility'], color='red', label='Filtered Out', s=10)
            plt.xlabel('tslm')
            plt.ylabel('ask_iv')
            plt.title(f"{one_datetime}")
            plt.legend()
            plt.show()
        
    pass


if __name__ == "__main__":
    main()
