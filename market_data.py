# AIO class for representation of market data and incremental access
import math
import os
import re

import pandas
import random
import time
import requests
from pymongo import MongoClient
from binance.client import Client
from requests.exceptions import HTTPError, ChunkedEncodingError

from utils import aggregate_ohlcv_window, get_minute_difference, get_dict_of_lists_slice

KAIKO_API_KEY = 'b4fda6c9cfa8c173209c61171bd1b4aa'
BINANCE_API_KEY = 'xE0pxPbbPXmarmETXqgBoZSAPba9nr5Wy3pfOjmoOUAK6A0VQ2a2DpK4wonrzJgj'
BINANCE_API_SECRET = 'NT6ZJ8SKa2bQHrvDcXpYNg3ThZsgOFUibzjeAJ7HBt3X0JZd6YkwkQDPTjhvoQbP'
MAX_DOC_SIZE = 150000
binance_klines_indices = {
    'open': 1,
    'high': 2,
    'low': 3,
    'close': 4,
    'volume': 5,
    'timestamps': 6
}


class HistoricalInstrument:
    """
    Interface for financial instrument OHLCV + ta data. self.data is of form:

    {
        '_id': str,
        'interval': str,
        'exchange_code': str,
        'class_name': str,
        'code': str,
        'timestamps', list,
        'open': list,
        'high': list,
        'low': list,
        'close': list,
        'volume': list
    }
    """

    def __init__(self, instrument, interval='1m', api_key=KAIKO_API_KEY):
        self.api_key = api_key
        self.data = instrument
        self.interval = interval
        self.data.update({
            "_id": self.__repr__(),
            "interval": self.interval,
        })

    def __repr__(self):
        return ':'.join([self.data['exchange_code'], self.data['class'], self.data['code'], self.interval])

    def to_dict(self):
        return self.data

    def load_binance_klines(self, csv_path):
        df = pandas.read_csv(csv_path)
        metrics = ['close_time', 'open', 'close', 'high', 'low', 'volume']
        for metric in metrics:
            save_metric = "timestamps" if metric == "close_time" else metric
            to_load = list(df[metric])
            if save_metric == "timestamps":
                print("Converting to timestamp...")
                to_load = [t * 1000 for t in to_load]
                print("Done conversion")
            self.data[save_metric] = [float(point) for point in to_load]
        self.data['size'] = len(self.data['open'])

    def request_kaiko_klines(self, interval='1m'):
        try:
            initial_res = requests.get(
                f'https://eu.market-api.kaiko.io/v1/data/trades.latest/exchanges/{self.data["exchange"]}/'
                f'{self.data["class"]}/{self.data["code"]}/aggregations/ohlcv',
                headers={f'X-Api-Key': self.api_key, 'Accept': 'application/json'},
                params={'interval': interval, 'page_size': 100000}
            )
            initial_res.raise_for_status()
        except (HTTPError, ChunkedEncodingError) as e:
            print(f"Retrying: {e}")
            time.sleep(5)
            return self.request_kaiko_klines(interval=interval)
        OHLCV = initial_res.json()['data']
        if 'next_url' in initial_res.json():
            next_url = initial_res.json()['next_url']
            while True:
                try:
                    next_res = requests.get(next_url,
                                            headers={f'X-Api-Key': self.api_key, 'Accept': 'application/json', },
                                            params={'interval': '1m'})
                    page_data = next_res.json()['data']
                    OHLCV += page_data
                    if 'next_url' in next_res.json():
                        next_url = next_res.json()['next_url']
                        continue
                    break
                except:
                    break
        for metric in ("timestamps", "open", "high", "low", "close", "volume"):
            self.data[metric] = [float(point[metric]) for point in OHLCV]
        print(f"{self}\tTotal size: {len(self.data['open'])}")

    def get_random_aggregated_ohlcv_windows(self, n_points=200, aggregates=(1,), minute_timestamps=True):
        print(f"{self}: Getting random windows")
        max_aggregate = max(aggregates)
        window_size = max_aggregate * n_points
        start_index = random.randint(0, len(self.data['open']) - window_size)
        end_index = start_index + window_size
        full_window = get_dict_of_lists_slice(self.data,
                                              start=start_index,
                                              end=end_index,
                                              keys=('timestamps', 'open', 'high', 'low', 'close', 'volume'))
        return {
            aggregate:
            aggregate_ohlcv_window(get_dict_of_lists_slice(full_window, start=window_size - n_points * aggregate),
                                   aggregate,
                                   minute_timestamps=minute_timestamps)
            for aggregate in aggregates
        }


class RealtimeInstrumentData:
    def __init__(self, symbol, window_size=10, keys=('timestamps', 'open', 'high', 'low', 'close', 'volume')):
        self.symbol = symbol
        self.window_size = window_size
        self.window = None
        self.__binance_client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
        self.keys = keys
        self.init_timestamp_unix = None

    def retrieve_latest_window(self):
        # TODO: Need to change to support multi-timescale convolutions
        return NotImplementedError
        last_timestep = self.window['timestamps'][-1]

        kline = self.__binance_client.get_klines(symbol=self.symbol, limit=1, interval='1m')[0]
        curr_timestamp = self.__get_relative_timestamp(kline[binance_klines_indices['timestamps']])

        if curr_timestamp == last_timestep:
            self.window['close'][-1] = self.get_latest_price()
            return self.window, False

        for k in self.keys:
            point = float(kline[binance_klines_indices[k]])
            if k == 'timestamps':
                self.window['timestamps'].append(self.__get_relative_timestamp(point))
            else:
                self.window[k].append(point)
        return self.window, True

    def init_window(self):
        klines = self.__binance_client.get_klines(symbol=self.symbol, limit=self.window_size, interval='1m')
        self.window = {k: [float(step[binance_klines_indices[k]]) for step in klines] for k in self.keys}
        self.init_timestamp_unix = self.window['timestamps'][0]
        self.window['timestamps'] = [self.__get_relative_timestamp(t) for t in self.window['timestamps']]

        return self.window

    def get_latest_price(self):
        return float(self.__binance_client.get_symbol_ticker(symbol=self.symbol)['price'])

    def __get_relative_timestamp(self, unix_timestamp):
        return get_minute_difference(self.init_timestamp_unix, unix_timestamp)


class HistoricalInstrumentDataset:
    def __init__(self, min_points=0, db_host="mongodb://localhost:27017/", api_key=KAIKO_API_KEY):
        self.api_key = api_key
        self.min_points = min_points
        self.exchanges = {}
        self.to_update = []
        self.train_ids, self.test_ids = [], []
        self.__mongo_client = MongoClient(db_host)
        self.__db = self.__mongo_client["crypto_rl"]
        self.__instrument_collection = self.__db["instruments"]
        self.__exchange_collection = self.__db["exchanges"]
        self.__split_collection = self.__db["splits"]
        self.load_split()

    def get_random_windows(self, n_points=200, minute_timestamps=True, split='test', aggregates=(1,)):
        instrument_id = random.choice(self.test_ids if split == 'test' else self.train_ids)
        instrument = HistoricalInstrument(self.__instrument_collection.find_one({'_id': instrument_id}))
        aggregated_windows = instrument.get_random_aggregated_ohlcv_windows(n_points=n_points, aggregates=aggregates,
                                                                            minute_timestamps=minute_timestamps)
        return aggregated_windows

    def get_instrument_ids(self):
        return self.__instrument_collection.distinct('_id', filter={f'open.{self.min_points}': {'$exists': True}})

    def create_split(self, train_prop=0.95):
        instrument_ids = self.get_instrument_ids()
        print(f"Number of instruments: {len(instrument_ids)}")
        random.shuffle(instrument_ids)
        i = int(train_prop * len(instrument_ids))
        split = {'train': instrument_ids[:i], 'test': instrument_ids[i:]}
        self.__split_collection.update_one({'_id': self.min_points}, {'$set': split}, upsert=True)
        return split

    def load_split(self):
        split = self.__split_collection.find_one({'_id': self.min_points})
        if split is None:
            split = self.create_split()
        self.train_ids, self.test_ids = split['train'], split['test']
        print(f"Number of train instruments: {len(self.train_ids)}\nNumber of test instruments: {len(self.test_ids)}")

    def request_exchanges(self):
        r = requests.get('https://reference-data-api.kaiko.io/v1/exchanges')
        self.exchanges = r.json()

    def save_all_kaiko_data(self, override_cache=False, interval='1m'):
        r = requests.get('https://reference-data-api.kaiko.io/v1/instruments')
        instruments = r.json()['data']
        for i, instrument_data in enumerate(instruments):
            instrument = HistoricalInstrument(instrument_data, interval=interval)
            instrument.request_kaiko_klines()
            self.save_instrument(instrument, override_cache=override_cache)
            if i % 100 == 0:
                print(f"Saving {i}/{len(instruments)}")

    def save_all_binance_data(self, data_dir='data', override_cache=True):
        for file in os.listdir(data_dir):
            filepath = os.path.join(data_dir, file)
            instrument_data = {
                'exchange_code': 'binanceapi',
                'code': re.search('[A-Z]+\-', file).group(0)[:-1].lower(),
                'class': 'option',
                'interval': '1m',
            }
            instrument = HistoricalInstrument(instrument_data)
            print(f"Loading csv for {file}")
            instrument.load_binance_klines(filepath)
            print(f"Saving {str(instrument)}")
            self.save_instrument(instrument, override_cache=override_cache)
            print("Succesfully saved, deleting CSV")
            os.remove(filepath)
            assert not os.path.exists(filepath)

    def save_instrument(self, instrument, override_cache=False, max_doc_size=MAX_DOC_SIZE):
        if self.__instrument_collection.find_one(
                {'_id': {"$regex": 'str(instrument).*'}}) is not None and not override_cache:
            print(f"{str(instrument)}: Already Cached")
        else:
            size = instrument.data['size']
            if size > max_doc_size:
                n_docs = math.ceil(size / max_doc_size)
                print(f"Size {size} larger than max- splitting into {n_docs} docs")
                for i in range(n_docs):
                    is_last = i == n_docs - 1
                    start = i * max_doc_size
                    end = (start + max_doc_size) if not is_last else size
                    _id = str(instrument) + f':{start}:{end}'
                    doc = {k: (v if not isinstance(v, list) else v[start:end]) for k, v in instrument.data.items()}
                    assert len(doc['open']) == (
                        max_doc_size if not is_last else size % max_doc_size), "You done fucked up lmao"
                    doc['_id'] = _id
                    doc['size'] = max_doc_size if not is_last else size % max_doc_size
                    self.__instrument_collection.update_one({'_id': _id}, {"$set": doc}, upsert=True)
            else:
                print("Size ok, saving")
                doc = instrument.to_dict()
                _id = str(instrument) + f'0:{size}'
                doc['_id'] = _id
                self.__instrument_collection.update_one({'_id': _id}, {"$set": instrument.to_dict()}, upsert=True)

    def get_instrument(self, id):
        return self.__instrument_collection.find({'_id': id})

    def update_data(self):
        pass


if __name__ == "__main__":
    hid = HistoricalInstrumentDataset(min_points=100000)
    print(hid.get_random_windows(n_points=10, aggregates=(1, 7, 10, 50)))
