import pandas as pd
import math
import os.path
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import timedelta, datetime
from dateutil import parser

BINANCE_API_KEY = 'xE0pxPbbPXmarmETXqgBoZSAPba9nr5Wy3pfOjmoOUAK6A0VQ2a2DpK4wonrzJgj'
BINANCE_API_SECRET = 'NT6ZJ8SKa2bQHrvDcXpYNg3ThZsgOFUibzjeAJ7HBt3X0JZd6YkwkQDPTjhvoQbP'

binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
binance_client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)


def minutes_of_new_data(symbol, kline_size, data):
    if len(data) > 0:
        old = parser.parse(data["timestamp"].iloc[-1])
    else:
        old = datetime.strptime('1 Jan 2017', '%d %b %Y')
    new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    return old, new


def get_all_binance(symbol, kline_size, save=False, ):
    filename = f'data/{symbol}-{kline_size}-data.csv'
    if os.path.isfile(filename):
        data_df = pd.read_csv(filename)
    else:
        data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df)
    delta_min = (newest_point - oldest_point).total_seconds() / 60
    available_data = math.ceil(delta_min / binsizes[kline_size])
    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'):
        print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))
    else:
        print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (
            delta_min, symbol, available_data, kline_size))
    time.sleep = lambda x: x
    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"),
                                                  newest_point.strftime("%d %b %Y %H:%M:%S"), limit=1000)
    data = pd.DataFrame(klines,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av',
                                 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else:
        data_df = data
    data_df.set_index('timestamp', inplace=True)
    if save:
        data_df.to_csv(filename)
    print('All caught up..!')
    return data_df


if __name__ == "__main__":
    print(len(os.listdir('data')))

    # info = binance_client.get_exchange_info()
    # start = 0
    # symbols = [sym['symbol'] for sym in info['symbols']]
    # print(len(symbols))
    # for sym in symbols:
    #     while True:
    #         try:
    #             get_all_binance(sym, '1m', save=True)
    #             break
    #         except BinanceAPIException:
    #             print("API Exception...")
    # get_all_binance('EVXETH', '1m', save=True)
