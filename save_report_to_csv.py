#!/usr/bin/env python
# coding: utf-8


# import pymongo
import datetime
import os

import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
import time
import json
import requests

# MONGODB_URI = "mongodb://heroku_8n7z0nqh:g0073297x@ds131313.mlab.com:31313/heroku_8n7z0nqh"

# client = MongoClient("mongodb://kai:g0073297x@ds131313.mlab.com:31313/heroku_8n7z0nqh")
# db = client['heroku_8n7z0nqh']

# client = MongoClient(host='ds131313.mlab.com',
#                      port=31313, 
#                      username='heroku_8n7z0nqh', 
#                      password='g0073297x',
#                     authSource="admin")
# db = client['heroku_8n7z0nqh']

# client = MongoClient(MONGODB_URI, connectTimeoutMS=30000)

# db = client["heroku_8n7z0nqh"]

# stocks_data = db.stocks_data

# import datetime 
# import time 

# xx = time.now()

# import time

# result = time.time()
# print("result:", result)
# print("\nyear:", result.tm_year)
# print("tm_hour:", result.tm_hour)

# xx = time.time()

# result = time.localtime(xx)
# print("result:", result)
# print("\nyear:", result.tm_year)
# print("tm_hour:", result.tm_hour)


# import time

# named_tuple = time.localtime() # get struct_time
# time_string = time.strftime("%d/%m/%Y", named_tuple)

# time_string


# import datetime
# d = datetime.datetime(2009, 3, 30)
# d

# from datetime import datetime

# datetime_str = '19/09/18'

# datetime_object = datetime.strptime(datetime_str, '%d/%m/%y')

# print(type(datetime_object))
# print(datetime_object)  # printed in default format

# connection = pymongo.MongoClient('ds131313.mlab.com', 31313)
# db = connection['heroku_8n7z0nqh']
# db.authenticate('kai', 'g0073297x')

# stocks_data = db['stocks_data']


# now = datetime.datetime.utcnow()
# last_2d = now - datetime.timedelta(days=2)

# stocks_data.delete_many({'utcdate': {"$lte": last_2d}})
# print(x.deleted_count, " documents deleted.")

# lisst = []

# for stock in stocks_data.find():
#     lisst = lisst + [stock]


# # READ FROM JSON

data = ''

import json

with open('../yahoofinances/stocks_data.json') as f:
    for h,i in enumerate(f): 
        i = i.strip()
        data += i

data = json.loads(data)


init_stock_sym = ['AAL', 'AAPL', 'ADBE', 'ADI', 'ADP', 'ADSK', 'ALGN', 'ALXN', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'ASML', 'ATVI', 'AVGO', 'BIDU', 'BIIB', 'BKNG', 'BMRN', 'CDNS', 'CELG', 
                    'CERN', 'CHKP', 'CHTR', 'CMCSA', 'COST', 'CSCO', 'CSX', 'CTAS', 'CTSH', 'CTXS', 'DLTR', 'EA', 'EBAY', 'EXPE', 'FAST', 'FB', 'FISV', 'FOX', 'FOXA', 'GILD', 
                    'GOOG', 'GOOGL', 'HAS', 'HSIC', 'IDXX', 'ILMN', 'INCY', 'INTC', 'INTU', 'ISRG', 'JBHT', 'JD', 'KHC', 'KLAC', 'LBTYA', 'LBTYK', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDLZ', 
                    'MELI', 'MNST', 'MSFT', 'MU', 'MXIM', 'MYL', 'NFLX', 'NTAP', 'NTES', 'NVDA', 'NXPI', 'ORLY', 'PAYX', 'PCAR', 'PEP', 'PYPL', 'QCOM', 'REGN', 'ROST', 'SBUX', 'SIRI', 
                    'SNPS', 'SWKS', 'TMUS', 'TSLA', 'TTWO', 'TXN', 'UAL', 'ULTA', 'VRSK', 'VRSN', 'VRTX', 'WBA', 'WDAY', 'WDC', 'WLTW', 'WYNN', 'XEL', 'XLNX']


sandp = ['TT','TFC','NOW','VIAC','PEAK','PAYC','BKR','ZBRA','STE','ODFL','J','WRB','NLOK','LYV','A','AAL','AAP','AAPL','ABBV','ABC','ABMD',
         'ABT','ACN','ADBE','ADI','ADM','ADP','ADS','ADSK','AEE','AEP','AES','AFL','AGN','AIG','AIV','AIZ','AJG','AKAM','ALB','ALGN','ALK','ALL','ALLE',
         'ALXN','AMAT','AMCR','AMD','AME','AMGN','AMP','AMT','AMZN','ANET','ANSS','ANTM','AON','AOS','APA','APD','APH','APTV','ARE','ARNC',
         'ATO','ATVI','AVB','AVGO','AVY','AWK','AXP','AZO','BA','BAC','BAX','BBY','BDX','BEN','BF-B','BIIB','BK','BKNG','BLK','BLL',
         'BMY','BR','BRK-B','BSX','BWA','BXP','C','CAG','CAH','CAT','CB','CBOE','CBRE','CCI','CCL','CDNS','CDW','CE','CERN','CF','CFG',
         'CHD','CHRW','CHTR','CI','CINF','CL','CLX','CMA','CMCSA','CME','CMG','CMI','CMS','CNC','CNP','COF','COG','COO','COP','COST','COTY','CPB','CPRI','CPRT',
         'CRM','CSCO','CSX','CTAS','CTL','CTSH','CTVA','CTXS','CVS','CVX','CXO','D','DAL','DD','DE','DFS','DG','DGX','DHI','DHR','DIS',
         'DISCA','DISCK','DISH','DLR','DLTR','DOV','DOW','DRE','DRI','DTE','DUK','DVA','DVN','DXC','EA','EBAY','ECL','ED','EFX','EIX',
         'EL','EMN','EMR','EOG','EQIX','EQR','ES','ESS','ETFC','ETN','ETR','EVRG','EW','EXC','EXPD','EXPE','EXR','F','FANG','FAST','FB','FBHS',
         'FCX','FDX','FE','FFIV','FIS','FISV','FITB','FLIR','FLS','FLT','FMC','FOX','FOXA','FRC','FRT','FTI','FTNT','FTV','GD','GE','GILD',
         'GIS','GL','GLW','GM','GOOG','GOOGL','GPC','GPN','GPS','GRMN','GS','GWW','HAL','HAS','HBAN','HBI','HCA','HD','HES','HFC','HIG','HII','HLT','HOG','HOLX',
         'HON','HP','HPE','HPQ','HRB','HRL','HSIC','HST','HSY','HUM','IBM','ICE','IDXX','IEX','IFF','ILMN','INCY','INFO','INTC','INTU',
         'IP','IPG','IPGP','IQV','IR','IRM','ISRG','IT','ITW','IVZ','JBHT','JCI','JKHY','JNJ','JNPR','JPM','JWN','K','KEY',
         'KEYS','KHC','KIM','KLAC','KMB','KMI','KMX','KO','KR','KSS','KSU','L','LB','LDOS','LEG','LEN','LH','LHX','LIN','LKQ','LLY','LMT',
         'LNC','LNT','LOW','LRCX','LUV','LVS','LW','LYB','M','MA','MAA','MAR','MAS','MCD','MCHP','MCK','MCO','MDLZ','MDT','MET','MGM','MHK','MKC',
         'MKTX','MLM','MMC','MMM','MNST','MO','MOS','MPC','MRK','MRO','MS','MSCI','MSFT','MSI','MTB','MTD','MU','MXIM','MYL','NBL','NCLH','NDAQ',
         'NEE','NEM','NFLX','NI','NKE','NLSN','NOC','NOV','NRG','NSC','NTAP','NTRS','NUE','NVDA','NVR','NWL','NWS','NWSA','O','OKE',
         'OMC','ORCL','ORLY','OXY','PAYX','PBCT','PCAR','PEG','PEP','PFE','PFG','PG','PGR','PH','PHM','PKG','PKI','PLD','PM','PNC',
         'PNR','PNW','PPG','PPL','PRGO','PRU','PSA','PSX','PVH','PWR','PXD','PYPL','QCOM','QRVO','RCL','RE','REG','REGN','RF','RHI',
         'RJF','RL','RMD','ROK','ROL','ROP','ROST','RSG','RTN','SBAC','SBUX','SCHW','SEE','SHW','SIVB','SJM','SLB','SLG','SNA',
         'SNPS','SO','SPG','SPGI','SRE','STT','STX','STZ','SWK','SWKS','SYF','SYK','SYY','T','TAP','TDG','TEL','TFX','TGT','TIF',
         'TJX','TMO','TMUS','TPR','TROW','TRV','TSCO','TSN','TTWO','TWTR','TXN','TXT','UA','UAA','UAL','UDR','UHS','ULTA','UNH','UNM',
         'UNP','UPS','URI','USB','UTX','V','VAR','VFC','VLO','VMC','VNO','VRSK','VRSN','VRTX','VTR','VZ','WAB','WAT','WBA','WDC','WEC',
         'WELL','WFC','WHR','WLTW','WM','WMB','WMT','WRK','WU','WY','WYNN','XEL','XLNX','XOM','XRAY','XRX','XYL','YUM','ZBH','ZION','ZTS']

total = init_stock_sym + sandp

set_total = (set(total))
list_set_total = list(set_total)

stocks = {}

for s in list_set_total: 
    for d in data: 
        if d['stock_symbol'] == s: 
            try:
#                 stocks[s].update(d)
                stocks[s].update({k:v for k,v in d.items() if v})

            except:
                stocks[s] = {}
                stocks[s].update(d)


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None



pd_stocks = pd.DataFrame(stocks).T


from yahooquery import Ticker

tickers = Ticker(list_set_total)

lissst = tickers.financial_data

df_analyst = pd.DataFrame(lissst).T[['currentPrice', 'targetHighPrice', 'targetLowPrice', 'targetMedianPrice', 'recommendationMean', 'recommendationKey', 'numberOfAnalystOpinions']]

df_analyst['currentPrice_divide_targetLowPrice'] = df_analyst['currentPrice'].values.astype(float) / np.array([float(i) if i != {} else 0 for i in df_analyst['targetLowPrice'].values])

df_analyst['currentPrice_divide_targetMedianPrice'] = df_analyst['currentPrice'].values.astype(float) / np.array([float(i) if i != {} else 0 for i in df_analyst['targetMedianPrice'].values])

df_analyst['currentPrice_divide_targetHighPrice'] = df_analyst['currentPrice'].values.astype(float) / np.array([float(i) if i != {} else 0 for i in df_analyst['targetHighPrice'].values])

refined_df_analyst = df_analyst[['recommendationMean', 'recommendationKey', 'numberOfAnalystOpinions', 'currentPrice_divide_targetLowPrice', 'currentPrice_divide_targetMedianPrice', 'currentPrice_divide_targetHighPrice']]

pd_stocks = pd.concat([pd_stocks, refined_df_analyst], axis=1)

# # Intrinsic Value [Without considering debt]


Growth_Rate_yrs1_5 = pd_stocks['Next 5 years (per annum)']

growth_1_5 = np.array([])

for i in Growth_Rate_yrs1_5: 
    try:
        if i[-1:] != '%': 
            i = 0 
            growth_1_5 = np.append(growth_1_5 ,i)
        else: 
            growth_1_5 = np.append(growth_1_5 ,i[:-1])
    except:
        growth_1_5 = np.append(growth_1_5,0)

growth_1_5

float_growth_1_5 = growth_1_5.astype(np.float)

float_growth_6_10 = float_growth_1_5 - 2

float_growth_1_5 = (float_growth_1_5/100) + 1

float_growth_6_10 = (float_growth_6_10/100) + 1

Year_1_Free_Cash_Flow = pd_stocks['Levered free cash flow (ttm)']

Year_1_Free_Cash_Flow

year_1_free_cash_flow = np.array([])

for i in Year_1_Free_Cash_Flow: 
    try:
        if i[-1:] != 'M' and i[-1:] != 'B': 
            i = 0 
            year_1_free_cash_flow = np.append(year_1_free_cash_flow ,i)
        else: 
            if i[-1:] == 'M': 
                year_1_free_cash_flow = np.append(year_1_free_cash_flow, float(i[:-1]) * 1000000)
            elif i[-1:] == 'B': 
                
                year_1_free_cash_flow = np.append(year_1_free_cash_flow , float(i[:-1]) * 1000000000)
    except:
        year_1_free_cash_flow = np.append(year_1_free_cash_flow,0)


year_1_free_cash_flow

excess_capital_cash = pd_stocks['Total Cash (mrq)']

excess_capital_cash

capital_cash = np.array([])

for i in excess_capital_cash: 
    try:
        if i[-1:] != 'M' and i[-1:] != 'B': 
            i = 0 
            capital_cash = np.append(capital_cash ,i)
        else: 
            if i[-1:] == 'M': 
                capital_cash = np.append(capital_cash, float(i[:-1]) * 1000000)
            elif i[-1:] == 'B': 
                
                capital_cash = np.append(capital_cash , float(i[:-1]) * 1000000000)
    except:
        capital_cash = np.append(capital_cash,0)

capital_cash

current_market_cap = pd_stocks['Market Cap (intraday)_current']

curr_market_cap = np.array([])

for i in current_market_cap: 
    try:
        if i[-1:] != 'M' and i[-1:] != 'B' and i[-1:] != 'T': 
            i = 0 
            curr_market_cap = np.append(curr_market_cap ,i)
        else: 
            if i[-1:] == 'M': 
                curr_market_cap = np.append(curr_market_cap, float(i[:-1]) * 1000000)
            elif i[-1:] == 'B': 
                
                curr_market_cap = np.append(curr_market_cap , float(i[:-1]) * 1000000000)
            elif i[-1:] == 'T':
                curr_market_cap = np.append(curr_market_cap , float(i[:-1]) * 1000000000000)

    except:
        curr_market_cap = np.append(curr_market_cap,0)


share_outstanding = pd_stocks['Shares Outstanding'] 

shares_outstanding = np.array([])

for i in share_outstanding: 
    try:
        if i[-1:] != 'M' and i[-1:] != 'B' and i[-1:] != 'T': 
            i = 0 
            shares_outstanding = np.append(shares_outstanding ,i)
        else: 
            if i[-1:] == 'M': 
                shares_outstanding = np.append(shares_outstanding, float(i[:-1]) * 1000000)
            elif i[-1:] == 'B': 
                
                shares_outstanding = np.append(shares_outstanding , float(i[:-1]) * 1000000000)
            elif i[-1:] == 'T':
                shares_outstanding = np.append(shares_outstanding , float(i[:-1]) * 1000000000000)

    except:
        shares_outstanding = np.append(shares_outstanding,0)


Terminal_Value_multipleofFCF = 15 
Discount_Rate = 10 # Assuming the discount rate (the return you expect to get)

calculation = {}
calculation['fcfyear1'] = year_1_free_cash_flow
calculation['fcfyear2'] = year_1_free_cash_flow * float_growth_1_5
calculation['fcfyear3'] = year_1_free_cash_flow * np.power(float_growth_1_5, 2)
calculation['fcfyear4'] = year_1_free_cash_flow * np.power(float_growth_1_5, 3)
calculation['fcfyear5'] = year_1_free_cash_flow * np.power(float_growth_1_5, 4)
calculation['fcfyear6'] = year_1_free_cash_flow * np.power(float_growth_1_5, 5)
calculation['fcfyear7'] = calculation['fcfyear6'] * np.power(float_growth_6_10, 1)
calculation['fcfyear8'] = calculation['fcfyear6'] * np.power(float_growth_6_10, 2)
calculation['fcfyear9'] = calculation['fcfyear6'] * np.power(float_growth_6_10, 3)
calculation['fcfyear10'] = calculation['fcfyear6'] * np.power(float_growth_6_10, 4)
calculation['fcfterminal'] = Terminal_Value_multipleofFCF * calculation['fcfyear10']


calculation['pvyear1'] = calculation['fcfyear1'] * (100 - Discount_Rate)/100
calculation['pvyear2'] = calculation['fcfyear2'] * np.power((100 - Discount_Rate)/100, 2)
calculation['pvyear3'] = calculation['fcfyear3'] * np.power((100 - Discount_Rate)/100, 3)
calculation['pvyear4'] = calculation['fcfyear4'] * np.power((100 - Discount_Rate)/100, 4)
calculation['pvyear5'] = calculation['fcfyear5'] * np.power((100 - Discount_Rate)/100, 5)
calculation['pvyear6'] = calculation['fcfyear6'] * np.power((100 - Discount_Rate)/100, 6)
calculation['pvyear7'] = calculation['fcfyear7'] * np.power((100 - Discount_Rate)/100, 7)
calculation['pvyear8'] = calculation['fcfyear8'] * np.power((100 - Discount_Rate)/100, 8)
calculation['pvyear9'] = calculation['fcfyear9'] * np.power((100 - Discount_Rate)/100, 9)
calculation['pvyear10'] = calculation['fcfyear10'] * np.power((100 - Discount_Rate)/100, 10)
calculation['pvterminal'] = calculation['fcfterminal'] * np.power((100 - Discount_Rate)/100, 10)


present_value_of_future_cashflows = calculation['pvyear1'] + calculation['pvyear2'] + calculation['pvyear3'] + calculation['pvyear4'] + calculation['pvyear5'] + calculation['pvyear6'] + calculation['pvyear7'] + calculation['pvyear8'] + calculation['pvyear9'] + calculation['pvyear10'] + calculation['pvterminal']

intrinsic_value_market_cap = present_value_of_future_cashflows + capital_cash

intrinsic_price_per_share = intrinsic_value_market_cap / (shares_outstanding)

price_per_share = curr_market_cap / shares_outstanding
price_per_share

pd_stocks['Intrinsic Market Cap(Mils)'] = intrinsic_value_market_cap

pd_stocks['Intrinsic Market Cap Per Share'] = intrinsic_price_per_share

pd_stocks['Market Cap Per Share'] = price_per_share


# # Intrinsic Value [Considering debt]

enterprise_value = pd_stocks['Enterprise Value_current']

enterprise_values = np.array([])

for i in enterprise_value: 
    try:
        if i[-1:] != 'M' and i[-1:] != 'B' and i[-1:] != 'T': 
            i = 0 
            enterprise_values = np.append(enterprise_values ,i)
        else: 
            if i[-1:] == 'M': 
                enterprise_values = np.append(enterprise_values, float(i[:-1]) * 1000000)
            elif i[-1:] == 'B': 
                
                enterprise_values = np.append(enterprise_values , float(i[:-1]) * 1000000000)
            elif i[-1:] == 'T':
                enterprise_values = np.append(enterprise_values , float(i[:-1]) * 1000000000000)

    except:
        enterprise_values = np.append(enterprise_values,0)


intrinsic_value_enterprise_value = present_value_of_future_cashflows

intrinsic_enterprise_value_price_per_share = intrinsic_value_enterprise_value / shares_outstanding 

enterprise_value_price_per_share = enterprise_values / shares_outstanding

pd_stocks['Intrinsic Enterprise Value(Mils)'] = present_value_of_future_cashflows

pd_stocks['Intrinsic Enterprise Value Per Share'] = intrinsic_enterprise_value_price_per_share

pd_stocks['Enterprise Value Per Share'] = enterprise_value_price_per_share

column_names = ["date", "stock_symbol","Previous Close","Market Cap (intraday)_current", "Intrinsic Market Cap(Mils)", "Market Cap Per Share", "Intrinsic Market Cap Per Share", "Enterprise Value_current", "Intrinsic Enterprise Value(Mils)", "Enterprise Value Per Share", "Intrinsic Enterprise Value Per Share", "Undervalued or Overvalued", "Fair Value",'recommendationMean', 'recommendationKey', 'numberOfAnalystOpinions', 'currentPrice_divide_targetLowPrice', 'currentPrice_divide_targetMedianPrice', 'currentPrice_divide_targetHighPrice']

df = pd_stocks[column_names]

pd.options.display.float_format = '{:20,.2f}'.format
df['Intrinsic Market Cap(Mils)'] = (df['Intrinsic Market Cap(Mils)']/1000000)

df['Intrinsic Enterprise Value(Mils)'] = (df['Intrinsic Enterprise Value(Mils)']/1000000)

df['Fair Value'] = df['Fair Value'].str.extract('(-?\d+)').astype(float)

df['recommendationMean'] = np.array([float(i) if i != {} else 5 for i in df['recommendationMean'].values])

neww_df = df[(df['recommendationMean'].astype(float) <= 2.0) | (df['Undervalued or Overvalued'] == 'Undervalued')].sort_values(by ='Fair Value' , ascending=False)


print(len(neww_df))

# neww_df.to_csv("daily_stock_analysis.csv", mode='a',sep=',')

pd_graph = pd.DataFrame(columns=['stock_symbol','RSI-4', 'RSI-3','RSI-2', 'RSI-1', 'RSI-now', 'MACD-4', 'MACD-3','MACD-2', 'MACD-1', 'MACD-now', 'MACD9-4', 'MACD9-3', 'MACD9-2', 'MACD9-1', 'MACD9-now', 'MACDhist-4', 'MACDhist-3', 'MACDhist-2', 'MACDhist-1', 'MACDhist-now'])

named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%d_%m_%Y", named_tuple)

dir_name = 'stock_chart_{}'.format(time_string)

os.mkdir(dir_name)


for i in neww_df['stock_symbol'].values.tolist():
    
    try: 
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

    except:
        time.sleep(13)
        continue

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
        plt.savefig('./{}/{}_{}_{}.png'.format(dir_name,stock_symbol, 'detail', last_date))
        plt.clf()
        plt.close()


    clean_data = clean_date(new_df, 'adjustedClose')

    create_plot(clean_data.astype(float), stock_symbol) # Graph for own observation 

    try: 
        START_DATE = df_date['date_data'].iloc[-252]
        END_DATE = df_date['date_data'].iloc[-1]
    except: 
        START_DATE = df_date['date_data'].iloc[0]
        END_DATE = df_date['date_data'].iloc[-1]

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
        plt.savefig('./{}/{}_{}_{}.png'.format(dir_name, stock_symbol, 'detail_1y', last_date))
        plt.clf()
        plt.close()



    clean_data = clean_date(new_df, 'adjustedClose')

    create_plot(clean_data.astype(float), stock_symbol) # Graph for own observation 

    def RSI(series, period):
        delta = series.diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
        u = u.drop(u.index[:(period-1)])
        d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
        d = d.drop(d.index[:(period-1)])
        rs = pd.Series.ewm(u, com=period-1, adjust=False).mean() / pd.Series.ewm(d, com=period-1, adjust=False).mean()
        return 100 - 100 / (1 + rs)
    
    pd_clean = pd.DataFrame(clean_data)

    pd_clean['RSI'] = RSI(clean_data.astype(float), 14)
    
    # create an MACD indicators
    def Plot_MACD(df, feature, fast=9, medium=12, long=26):
        # WARNING - Feed data in ascending order only (i.e. first row should be your oldest print)
        
        tmp_df = df.copy()
        
        # Price chart
    #     fig, ax = plt.subplots(figsize=(16, 8))
        tmp_df['EXP1'] = tmp_df[feature].ewm(span=medium, adjust=False).mean()
        tmp_df['EXP2'] = tmp_df[feature].ewm(span=long, adjust=False).mean()
        tmp_df['MACD'] = tmp_df['EXP1']-tmp_df['EXP2']
        tmp_df['EXP3'] = tmp_df['MACD'].ewm(span=fast, adjust=False).mean()
        tmp_df['MACD_Hist'] = tmp_df['MACD'] - tmp_df['EXP3']

    #     plt.plot(df.index, tmp_df[feature].astype(float), label='Price', color='black')
    #     plt.plot(df.index, tmp_df['EXP1'], label='EXP1-12', color='blue')
    #     plt.plot(df.index, tmp_df['EXP2'], label='EXP2-26', color='red')
    #     plt.title('MACD - ' + str(feature))
    #     plt.legend(loc='upper left')
    #     plt.grid()
    #     plt.show()

        df['RSI'].astype(float).plot(figsize=(15,9))
        plt.savefig('./{}/{}_{}_{}.png'.format(dir_name, stock_symbol, 'RSI_1y', last_date))
        plt.clf()
        plt.close()
        
        # Histogram chart
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.bar(df.index, tmp_df['MACD_Hist'] , width=1, label='Hist')
        df['MACD_Hist'] = tmp_df['MACD_Hist']
        ax.xaxis_date
        plt.plot(df.index,  tmp_df['MACD'], label='MACD',color='blue')
        df['MACD'] = tmp_df['MACD']
        plt.plot(df.index,  tmp_df['EXP3'], label='MACD-9',color='red')
        df['MACD-9'] = tmp_df['EXP3']
        plt.axhline(0, color='gray', linewidth=3, linestyle='-.' )
        plt.legend(loc='upper left')
        plt.grid()
        plt.savefig('./{}/{}_{}_{}.png'.format(dir_name, stock_symbol, 'MACD_1y', last_date))
        plt.clf()
        plt.close()
    
    Plot_MACD(pd_clean, 'adjustedClose')

    # Calculate Annualized return 

    START_DATE = df_date['date_data'].iloc[0]
    END_DATE = df_date['date_data'].iloc[-1]

    clean_data = clean_date(new_df, 'adjustedClose')

    # new_df
    # Since inception 
    print(clean_data)
    no_years = ((clean_data.index[-1] - clean_data.index[0]).days)/365
    print(no_years)
    annualized_return_since_inception = ((np.power((float(clean_data[-1])/float(clean_data[0])), 1/no_years)) - 1)*100
    print(annualized_return_since_inception) 


    # Past 10 years stock annualized return 
    try: 
        annualized_return_since_past10years = ((np.power((float(clean_data[-1])/float(clean_data[-3650])), 1/10)) - 1)*100
    except: 
        annualized_return_since_past10years = None

    # Past 5 years stock annualized return 
    try:
        annualized_return_since_past5years = ((np.power((float(clean_data[-1])/float(clean_data[-1825])), 1/5)) - 1)*100
    except: 
        annualized_return_since_past5years = None

    # Calculate Annulized volatility 

    # Standard Deviation since inception 
    new_df['daily_change'] = new_df['adjustedClose'].astype(float).diff()
    new_df['adjustedCloseShift'] = new_df['adjustedClose'].shift()
    new_df['percentage_daily_change'] = (new_df['daily_change'].astype(float) / new_df['adjustedCloseShift'].astype(float)) * 100
    std_percent = new_df['percentage_daily_change'].std()
    annualized_std = std_percent * np.sqrt(252)

    # ten years standard deviation 
    try: 
        years_10_new_df = new_df.iloc[-2520:]
        std_percent_10years = years_10_new_df['percentage_daily_change'].std()
        annualized_10_std = std_percent_10years * np.sqrt(252)
    except: 
        annualized_10_std = None

    # Five years standard deviation 
    try: 
        years_5_new_df = new_df.iloc[-1260:]
        std_percent_5years = years_5_new_df['percentage_daily_change'].std()
        annualized_5_std = std_percent_5years * np.sqrt(252)
    except: 
        annualized_5_std = None


    pd_graph = pd_graph.append({'stock_symbol': stock_symbol.upper(),'Annualized_return_since_incep': annualized_return_since_inception, 'Annualized_return_past10years': annualized_return_since_past10years, 'Annualized_return_past5years': annualized_return_since_past5years, 'Annualized_std_since_incep': annualized_std, 'Annualized_std_since_past10years': annualized_10_std, 'Annualized_std_since_past5years': annualized_5_std ,'RSI-4': pd_clean['RSI'][-5],'RSI-3': pd_clean['RSI'][-4],'RSI-2': pd_clean['RSI'][-3],'RSI-1': pd_clean['RSI'][-2],'RSI-now': pd_clean['RSI'][-1],'MACD-4': pd_clean['MACD'][-5],'MACD-3': pd_clean['MACD'][-4],'MACD-2': pd_clean['MACD'][-3],'MACD-1': pd_clean['MACD'][-2],'MACD-now': pd_clean['MACD'][-1],'MACD9-4': pd_clean['MACD-9'][-5],'MACD9-3': pd_clean['MACD-9'][-4],'MACD9-2': pd_clean['MACD-9'][-3],'MACD9-1': pd_clean['MACD-9'][-2],'MACD9-now': pd_clean['MACD-9'][-1],'MACDhist-4': pd_clean['MACD_Hist'][-5],'MACDhist-3': pd_clean['MACD_Hist'][-4],'MACDhist-2': pd_clean['MACD_Hist'][-3],'MACDhist-1': pd_clean['MACD_Hist'][-2],'MACDhist-now': pd_clean['MACD_Hist'][-1]}, ignore_index=True)
    

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

    time.sleep(13)

neww_df = neww_df.merge(pd_graph, left_on='stock_symbol', right_on='stock_symbol')

neww_df.to_csv("daily_stock_analysis.csv", mode='a',sep=',')




# # Below is for illustration only (Not for production) 


# Apple stock example

# Growth_Rate_yrs1_5 = 1.1128 # in percentage terms actually
# Growth_Rate_yrs6_10 = 1.09 #Assuming that growth rate is reduced by 1 percent
# Discount_Rate = 10 # Assuming the discount rate (the return you expect to get)
# Terminal_Value_multipleofFCF = 15 
# Year_1_Free_Cash_Flow = 45.59 * 1000000000  # Multiple by 1 billion 
# Excess_Capital_Cash = 107.16 * 1000000000  

# current_market_cap = 1.13 * 1000000000000
# shares_outstanding = 4.38 * 1000000000  



# calculation = {}
# calculation['fcfyear1'] = Year_1_Free_Cash_Flow
# calculation['fcfyear2'] = Year_1_Free_Cash_Flow * Growth_Rate_yrs1_5
# calculation['fcfyear3'] = Year_1_Free_Cash_Flow * np.power(Growth_Rate_yrs1_5, 2)
# calculation['fcfyear4'] = Year_1_Free_Cash_Flow * np.power(Growth_Rate_yrs1_5, 3)
# calculation['fcfyear5'] = Year_1_Free_Cash_Flow * np.power(Growth_Rate_yrs1_5, 4)
# calculation['fcfyear6'] = Year_1_Free_Cash_Flow * np.power(Growth_Rate_yrs1_5, 5)
# calculation['fcfyear7'] = calculation['fcfyear6'] * np.power(Growth_Rate_yrs6_10, 1)
# calculation['fcfyear8'] = calculation['fcfyear6'] * np.power(Growth_Rate_yrs6_10, 2)
# calculation['fcfyear9'] = calculation['fcfyear6'] * np.power(Growth_Rate_yrs6_10, 3)
# calculation['fcfyear10'] = calculation['fcfyear6'] * np.power(Growth_Rate_yrs6_10, 4)
# calculation['fcfterminal'] = Terminal_Value_multipleofFCF * calculation['fcfyear10']




# calculation['pvyear1'] = calculation['fcfyear1'] * (100 - Discount_Rate)/100
# calculation['pvyear2'] = calculation['fcfyear2'] * np.power((100 - Discount_Rate)/100, 2)
# calculation['pvyear3'] = calculation['fcfyear3'] * np.power((100 - Discount_Rate)/100, 3)
# calculation['pvyear4'] = calculation['fcfyear4'] * np.power((100 - Discount_Rate)/100, 4)
# calculation['pvyear5'] = calculation['fcfyear5'] * np.power((100 - Discount_Rate)/100, 5)
# calculation['pvyear6'] = calculation['fcfyear6'] * np.power((100 - Discount_Rate)/100, 6)
# calculation['pvyear7'] = calculation['fcfyear7'] * np.power((100 - Discount_Rate)/100, 7)
# calculation['pvyear8'] = calculation['fcfyear8'] * np.power((100 - Discount_Rate)/100, 8)
# calculation['pvyear9'] = calculation['fcfyear9'] * np.power((100 - Discount_Rate)/100, 9)
# calculation['pvyear10'] = calculation['fcfyear10'] * np.power((100 - Discount_Rate)/100, 10)
# calculation['pvterminal'] = calculation['fcfterminal'] * np.power((100 - Discount_Rate)/100, 10)



# present_value_of_future_cashflows = calculation['pvyear1'] + calculation['pvyear2'] + calculation['pvyear3'] + calculation['pvyear4'] + calculation['pvyear5'] + calculation['pvyear6'] + calculation['pvyear7'] + calculation['pvyear8'] + calculation['pvyear9'] + calculation['pvyear10'] + calculation['pvterminal']

# intrinsic_value_market_cap = present_value_of_future_cashflows + Excess_Capital_Cash

# calculation

# present_value_of_future_cashflows

# intrinsic_value_market_cap 

# intrinsic_value_market_cap / 1000000000000


# diff = current_market_cap - intrinsic_value_market_cap

# diff/1000000000

# intrinsic_price_per_share = intrinsic_value_market_cap / (shares_outstanding)

# price_per_share = current_market_cap / shares_outstanding

# price_per_share

# # # Intrinsic Value [Considering debt]

# Growth_Rate_yrs1_5 = 1.1128 # in percentage terms actually
# Growth_Rate_yrs6_10 = 1.09 #Assuming that growth rate is reduced by 1 percent
# Discount_Rate = 10 # Assuming the discount rate (the return you expect to get)
# Terminal_Value_multipleofFCF = 15 
# Year_1_Free_Cash_Flow = 45.59 * 1000000000  # Multiple by 1 billion

# enterprise_value = 1.13 * 1000000000000
# shares_outstanding = 4.38 * 1000000000  

# intrinsic_value_enterprise_value = present_value_of_future_cashflows

# diff_en = enterprise_value - intrinsic_value_enterprise_value

# diff_en 

# intrinsic_enterprise_value_price_per_share = intrinsic_value_enterprise_value / shares_outstanding 

# intrinsic_enterprise_value_price_per_share

# enterprise_value_price_per_share = enterprise_value / shares_outstanding

# enterprise_value_price_per_share

# from datetime import timedelta, datetime

# today = datetime.now()

# lastmonth = today - timedelta(days=30) # last 30 days 

# today

# lastmonth

# for date in tests:
#     if date >= str(lastmonth):
#         print(date)
# else:
#     pass





