from sec_edgar_downloader import Downloader

from config import ORIGINAL_DATA_DIR

TICKERS = ...  # TODO: fill in 10 sampled tickers



import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
from config import ORIGINAL_DATA_DIR

def download_10k_filings():
    """
    Downloads 10-K filings for all companies in sampled_tickers.txt from 2010-2020.
    Uses SEC EDGAR to fetch the filings and saves them to ORIGINAL_DATA_DIR.
    """
    
    with open('sampled_tickers.txt', 'r') as f:
        tickers = [line.strip() for line in f]
    
    for ticker in tickers:
        folder = ORIGINAL_DATA_DIR
        # print(folder)
        dl = Downloader("GT", "edutensor@gmail.com", folder)
        print(f"Downloading 10-K filings for {ticker}...")
        try:
            dl.get("10-K", ticker, 
                   after="2010-01-01",
                   before="2020-01-01",
                   download_details=True)
            # break
        except Exception as e:
            print(f"Error: Missing data for company {ticker} between 2010-2020")
            print(f"Error details: {str(e)}")



download_10k_filings()

# TODO: download 10-Ks for each ticker
# to the ORIGINAL_DATA_DIR directory.

# Possible scores:
# [5 pts]   All the 10-Ks available are downloaded
#           for all 10 tickers.
# [2.5 pts] Some of the data is missing.
# [0 pts]   No data is downloaded and stored.


