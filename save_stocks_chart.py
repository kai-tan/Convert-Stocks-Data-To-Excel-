#!/usr/bin/env python
# coding: utf-8

# # Changing to Percentage Return, using adj close instead. And predicting 3 months Stocks horizon instead of daily. I have added other stock fundamentals ratios as well in the hope that it will improve its prediction power. 

# I have used RapidAPI's alpha vantage and morningstar in order to get the data. 

# %tensorflow_version 2.x # Only for use in Colab 


import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
import time
import json
import datetime
import requests

from tensorflow.python.keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import regularizers


# from tensorflow.python.client import device_lib # Just to check whether I'm using CPU or GPU 
# print(device_lib.list_local_devices())


stocks_list = ['aal','aapl','adbe','adi','adp','adsk','algn','alxn','amat','amd','amgn','amzn','asml','atvi','avgo','bidu','biib','bkng'
            ,'bmrn','cdns','celg','cern','chkp','chtr','cmcsa','cost','csco','csx','ctas','ctrp','ctsh','ctxs','dltr','ea','ebay'
            ,'expe','fast','fb','fisv','fox','foxa','gild','goog','googl','has','hsic','idxx','ilmn','incy','intc','intu','isrg','jbht'
               ,'jd','khc','klac','lbtya','lbtyk','lrcx','lulu','mar','mchp','mdlz','meli','mnst','msft','mu','mxim','myl','nflx','ntap'
               ,'ntes','nvda','nxpi','orly','payx','pcar','pep','pypl','qcom','regn','rost','sbux','siri','snps','swks','symc','tmus',
               'tsla','ttwo','txn','ual','ulta','vrsk','vrsn','vrtx','wba','wday','wdc','wltw','wynn','xel','xlnx']

SandPplusNasdaq = ['DVA', 'ETFC', 'TFC', 'BBY', 'AMP', 'AVB', 'PNC', 'RSG', 'AVY', 'JCI', 'CRM', 'SBAC', 'BAX', 'CTSH', 'INCY', 'FDX', 'O', 'EQR', 'CMCSA', 'AMT', 'FLIR', 'UHS', 'PKI', 
                    'TSLA', 'IBM', 'PFG', 'QRVO', 'JPM', 'MO', 'LMT', 'TSN', 'NI', 'ZTS', 'CFG', 'SYMC', 'MSI', 'MSCI', 'RF', 'GOOG', 'BKNG', 'CSX', 'KLAC', 'LULU', 'BWA', 'ULTA', 'CXO', 
                    'AIG', 'PG', 'VRSN', 'GE', 'TJX', 'AKAM', 'XOM', 'C', 'FRC', 'SYK', 'CMS', 'CBRE', 'LW', 'DRE', 'TIF', 'BXP', 'AEE', 'ADM', 'GPC', 'RHI', 'XRAY', 'SYY', 'HSY', 'WRB', 
                    'COO', 'CVX', 'APTV', 'ARNC', 'NLSN', 'SEE', 'ISRG', 'MDT', 'GWW', 'ABBV', 'WAB', 'PNR', 'ASML', 'KHC', 'JNJ', 'UNM', 'BA', 'HLT', 'EMN', 'NVDA', 'MCK', 'GD', 'AAL', 
                    'FBHS', 'ESS', 'DISCA', 'MA', 'NOW', 'FITB', 'GM', 'LNT', 'WELL', 'SLG', 'KSU', 'IT', 'A', 'NWSA', 'UNH', 'ZBRA', 'PAYX', 'M', 'URI', 'VMC', 'SIRI', 'XYL', 'FLS', 
                    'MCD', 'ALK', 'EXPD', 'ALB', 'CINF', 'FTV', 'CHD', 'AXP', 'SNA', 'AFL', 'CCI', 'PHM', 'IRM', 'MTD', 'MAR', 'CF', 'DIS', 'DFS', 'EIX', 'HPQ', 'K', 'VNO', 'CAG', 'DOW', 
                    'IFF', 'RCL', 'MSFT', 'HBAN', 'AMCR', 'NCLH', 'XLNX', 'WAT', 'DGX', 'SIVB', 'AEP', 'HCA', 'PCAR', 'PSX', 'BDX', 'HAL', 'WBA', 'UPS', 'COF', 'LH', 'MHK', 'KEYS', 'ALGN', 
                    'BKR', 'FLT', 'PM', 'NOV', 'PNW', 'HES', 'IEX', 'LOW', 'NEM', 'SO', 'ADP', 'REG', 'CERN', 'FB', 'PWR', 'AVGO', 'TXN', 'QCOM', 'ALLE', 'USB', 'BMRN', 'MMM', 'ZBH', 'MS', 
                    'AMZN', 'VIAC', 'ECL', 'PPL', 'WDAY', 'NLOK', 'MU', 'V', 'CLX', 'FANG', 'WLTW', 'BR', 'NKE', 'NWL', 'EW', 'KO', 'KEY', 'WFC', 'LIN', 'AIV', 'RJF', 'ADBE', 'IPGP', 'ROL', 
                    'CPRI', 'DOV', 'SPGI', 'BRK-B', 'DRI', 'DE', 'EOG', 'CHRW', 'NBL', 'WMB', 'T', 'AOS', 'BLK', 'MXIM', 'VRSK', 'CB', 'TGT', 'WMT', 'ADSK', 'APA', 'AMGN', 'AMAT', 'WHR', 
                    'TXT', 'CNP', 'ICE', 'UNP', 'UAA', 'CTXS', 'ADS', 'PEG', 'MPC', 'AGN', 'ACN', 'WDC', 'FTI', 'OMC', 'AIZ', 'KR', 'CTL', 'DAL', 'CHKP', 'YUM', 'MGM', 'MELI', 'CNC', 'CVS', 
                    'CAT', 'PXD', 'DISH', 'HUM', 'MNST', 'PRU', 'HST', 'TTWO', 'AJG', 'IQV', 'WRK', 'OKE', 'TWTR', 'KMX', 'FOX', 'IVZ', 'VLO', 'CTVA', 'HBI', 'IDXX', 'RMD', 'ABC', 'OXY', 
                    'ANTM', 'MCO', 'WYNN', 'LNC', 'LB', 'VRTX', 'ABMD', 'LUV', 'COP', 'DHI', 'MMC', 'CI', 'NEE', 'GPS', 'MLM', 'PKG', 'COTY', 'ED', 'BSX', 'DTE', 'TSCO', 'SYF', 'SJM', 
                    'BEN', 'EXR', 'SWKS', 'VAR', 'AWK', 'KMB', 'HP', 'GOOGL', 'AAPL', 'NTAP', 'AME', 'TMUS', 'ROP', 'BK', 'DHR', 'ADI', 'KMI', 'TAP', 'AMD', 'J', 'INTU', 'DXC', 'TDG', 
                    'ARE', 'LHX', 'ATO', 'GILD', 'ORCL', 'WU', 'HAS', 'HFC', 'FIS', 'RTN', 'HOLX', 'HSIC', 'SWK', 'SCHW', 'CHTR', 'MET', 'RE', 'HIG', 'MKC', 'CPRT', 'EXC', 'PEP', 'FMC', 
                    'EL', 'COST', 'NSC', 'BIDU', 'PH', 'HRB', 'XEL', 'CBOE', 'PPG', 'AZO', 'JWN', 'INTC', 'LYB', 'PVH', 'NXPI', 'TROW', 'REGN', 'CL', 'ZION', 'NTRS', 'ABT', 'FE', 'BAC', 
                    'VFC', 'CDW', 'MRK', 'MCHP', 'HRL', 'CPB', 'BMY', 'TT', 'KSS', 'EFX', 'EQIX', 'LBTYA', 'ETN', 'PFE', 'PSA', 'LEN', 'CTRP', 'ANSS', 'XRX', 'SLB', 'ALXN', 'NDAQ', 'EXPE', 
                    'PEAK', 'JBHT', 'DLR', 'HD', 'WEC', 'ALL', 'CE', 'AES', 'TPR', 'HPE', 'NOC', 'PLD', 'FAST', 'CELG', 'D', 'NWS', 'GPN', 'PGR', 'TRV', 'FCX', 'PYPL', 'MRO', 'ATVI', 'ES', 
                    'GIS', 'FOXA', 'TMO', 'EA', 'VTR', 'NTES', 'FTNT', 'IP', 'APD', 'LYV', 'CDNS', 'IPG', 'EMR', 'UDR', 'GRMN', 'DUK', 'MKTX', 'MDLZ', 'PRGO', 'DLTR', 'L', 'WY', 'MYL', 
                    'AAP', 'AON', 'LRCX', 'FRT', 'NRG', 'FISV', 'SBUX', 'APH', 'JNPR', 'EVRG', 'F', 'UA', 'NVR', 'UTX', 'SHW', 'STT', 'TFX', 'GLW', 'LDOS', 'ITW', 'NUE', 'CSCO', 'BLL', 'LLY', 
                    'BF-B', 'UAL', 'IR', 'SRE', 'ETR', 'HII', 'WM', 'MOS', 'STX', 'EBAY', 'CAH', 'DD', 'ORLY', 'CCL', 'DG', 'ODFL', 'ROK', 'COG', 'HON', 'STE', 'ILMN', 'LKQ', 'GL', 'LEG', 
                    'VZ', 'LVS', 'CMG', 'GS', 'BIIB', 'STZ', 'CME', 'LBTYK', 'CTAS', 'HOG', 'JKHY', 'NFLX', 'CMA', 'MAA', 'SPG', 'ROST', 'FFIV', 'PBCT', 'RL', 'ANET', 'INFO', 'MAS', 'JD', 
                    'DISCK', 'MTB', 'DVN', 'PAYC', 'TEL', 'KIM', 'CMI', 'SNPS']


for i in SandPplusNasdaq:
    
    url = "https://alpha-vantage.p.rapidapi.com/query"

    querystring = {"outputsize":"full","datatype":"json","function":"TIME_SERIES_DAILY_ADJUSTED","symbol":'{}'.format(i)}

    headers = {
        'x-rapidapi-host': "alpha-vantage.p.rapidapi.com",
        'x-rapidapi-key': "622cc4aea0msh1ef679db027bf3dp12f333jsn5670556c4401"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)

    result_dict = json.loads(response.text)

    stock_symbol = result_dict['Meta Data']['2. Symbol']
    last_date = result_dict['Meta Data']['3. Last Refreshed']

    date_data = []
    high = []
    low = []
    adjustedClose = []
    volume = []

    for i, j in result_dict['Time Series (Daily)'].items():

      date_data.append(i)

      for k, l in j.items():
        
        if k == '2. high': 
          high.append(l)
        elif k == '3. low':
          low.append(l)
        elif k == '5. adjusted close': 
          adjustedClose.append(l)
        elif k == '6. volume': 
          volume.append(l)



    df_date = pd.DataFrame({ 'date_data': date_data, 'high': high, 'low': low, 
    'adjustedClose': adjustedClose, 'volume': volume })

    df_date['date_data'] =  pd.to_datetime(df_date['date_data'], format='%Y-%m-%d')

    df_date['year'] = df_date['date_data'].dt.year

    df_date = df_date.iloc[::-1].reset_index(drop=True)

    new_df = df_date.reindex(df_date['date_data'])

    new_df['adjustedClose'] = df_date['adjustedClose'].values


    START_DATE = df_date['date_data'].iloc[0]
    END_DATE = df_date['date_data'].iloc[-1]


    def clean_date(stock_data, col): 
        weekdays = pd.date_range(start=START_DATE, end=END_DATE)
        clean_data = stock_data[col].reindex(weekdays)
        return clean_data.fillna(method='ffill')


    def get_stats(stock_data): 
        return { 
            'short_rolling': stock_data.rolling(window=20).mean(),
            'medium_rolling': stock_data.rolling(window=50).mean(),
            'long_rolling': stock_data.rolling(window=200).mean()
        }

    def create_plot(stock_data, ticker): 
        plt.style.use('fivethirtyeight')
        stats = get_stats(stock_data)
        plt.subplots(figsize=(15,10))
        plt.plot(stock_data, label=ticker, linewidth=0.6)
        plt.plot(stats['short_rolling'], label='20 day rolling mean', linewidth=0.6)
        plt.plot(stats['medium_rolling'], label='50 days rolling mean', linewidth=0.6)
        plt.plot(stats['long_rolling'], label='200 day rolling mean', linewidth=0.6)
        plt.xlabel('Date')
        plt.ylabel('Adj Close (p)')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', linewidth=1)
        plt.title('Stock Price over Time.')
        plt.savefig('./Stock_Detail/{}_{}_{}.png'.format(stock_symbol, 'detail', last_date))
        plt.clf()
        plt.close()


    clean_data = clean_date(new_df, 'adjustedClose')

    create_plot(clean_data.astype(float), stock_symbol) # Graph for own observation 

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.style.use('ggplot')
    # plt.xticks([])
    # plt.yticks([])
    # plt.grid(False)
    # # X AXIS -BORDER
    # ax.spines['bottom'].set_visible(False)
    # # BLUE
    # ax.set_xticklabels([])
    # # RED
    # ax.set_xticks([])
    # # RED AND BLUE TOGETHER
    # ax.axes.get_xaxis().set_visible(False)
    # clean_data.astype(float).plot(linewidth=0.6)
    # plt.savefig('./Stock_AI/{}_{}_{}.png'.format(stock_symbol, 'ai', last_date))
    # plt.clf()
    # plt.close()

    time.sleep(15)



