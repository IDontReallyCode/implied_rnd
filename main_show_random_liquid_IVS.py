import os
import random
import polars as pl
import matplotlib.pyplot as plt

"""
The objective is simply to show visually a set of random Implied Volatility Surfaces (IVSs).

2024-01-02: This is a work in progress.  For now, this will simply show a truly random set of IVSs, selecting randomly the ticker, and selecting randomly the days.
"""

def main():
    # Set the seed for random number generation
    random.seed(84)

    directory = "E:/CBOE/ipc_per_day"
    N = 10  # Number of files to select

    file_list = os.listdir(directory)
    selected_files = random.sample(file_list, N)

    for file_name in selected_files:
        file_path = os.path.join(directory, file_name)
        df = pl.read_ipc(file_path)
        # Process the DataFrame as needed
        
        # Print the header for all columns
        print("Column Header:")
        print(df.columns)
        
        # pick a random value from the quote_datetime column
        random_value = random.choice(df['quote_datetime'])
        random_underlying_symbol = random.choice(df['underlying_symbol'])
        # print("Random value:", random_value)
        
        # filter the DataFrame to only include rows where the quote_datetime column is equal to the random value
        filtered_df = df.filter((pl.col('quote_datetime') == random_value) & (pl.col('otm') == True) & (pl.col('underlying_symbol') == random_underlying_symbol)
                                 & (pl.col('ask_iv') > 0.0))
        
        # Create a new plot with 'tslm' on x-axis and 'ask_iv' on y-axis
        plt.scatter(filtered_df['tslm'], filtered_df['ask_iv'])
        plt.xlabel('tslm')
        plt.ylabel('ask_iv')
        plt.title(f"{random_underlying_symbol} - {random_value}")  # Set the title as underlying_symbol and quote_datetime joined
        plt.show()
        
        stopdebug=1
        
    pass


if __name__ == "__main__":
    main()
