#### Settings
# Imports for data retrieval and analysis
import numpy as np
import pandas as pd
import yfinance
from pandas_datareader import data

import warnings
warnings.filterwarnings("ignore")

#choose the GAN:
#Choice of network: start with SAGAN (and YLGAN)
#Choice of loss function: Use hinge losses (standard one in most GANs), if mode collapse than change loss fct to wgan with gp

#choice = 'wgan_gp'

# defined in "Setup", copied here for simplicity
"""
BUFFER_SIZE = 5032 # lenght of timeseries 60k
BATCH_SIZE = 32 #256
data_dim = 32 #256
noise_dim = 100
data_channel = 1
"""

# Downloading the data from Yahooo finance
def download_data(start_date, end_date, ticker_list, name_list, ohlc_data):
    """
    Function to enter start and end date for specific security data

    :param start_date: 'dd-mm-yyyy', string
    :param end_date: 'dd-mm-yyyy', string
    :param ticker_list: list containing yahoo ticker codes as strings
    :param name_list: list containing ticker names as strings

    :return: dataframe of all adjusted closing prices
    """

    closing_data = pd.DataFrame()
    a = 0

    for name in name_list:
        closing_data[name_list[a]] = yfinance.download(
            ticker_list[a], start_date, end_date)[ohlc_data]
        a += 1

    return closing_data.fillna(method='ffill')



# Get Log Returns and store in dataframes
def f_log_return(data):
    tmp = np.asarray(data)
    n = tmp.shape[1]
    lst = [np.zeros(n)]
    for i in range(1,len(data)):
        lst.append(np.log(tmp[i]/tmp[i-1])*100)
    lst = np.array(lst)
    for i in range(n):
        data[f"Log Returns {i}"] = lst[:,i]
    return data



# Rolling window applied to dataframe before converting it to tensor
def rolling_window(data, window=32, stride=1):
    """
    takes an array and returns rolling window of inputs

    :data: array of values
    :window: size of window you would like each return input
    :stride: do we want to stride inputs

    :return: array of windows
    """
    x = []
    current = 0
    total = (len(data)/stride)-window
    for i in range(int(total)+1):
        x.append(data[current:current+window])
        current += stride
    return x