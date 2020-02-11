# AIO class for representation of market data and incremental access
import json

import pandas
import pymongo
import random
import time
import requests
import ta
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
from requests.exceptions import HTTPError, ChunkedEncodingError

API_KEY = 'b4fda6c9cfa8c173209c61171bd1b4aa'


class KaikoInstrument:
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
    def __init__(self, instrument, interval='1m', api_key=API_KEY):
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

    def request_OHLCV(self, interval='1m'):
        try:
            initial_res = requests.get(
                f'https://eu.market-api.kaiko.io/v1/data/trades.latest/exchanges/{self.data["exchange"]}/{self.data["class"]}/{self.data["code"]}/aggregations/ohlcv',
                headers={f'X-Api-Key': self.api_key, 'Accept': 'application/json'},
                params={'interval': interval, 'page_size': 100000}
            )
            initial_res.raise_for_status()
        except (HTTPError, ChunkedEncodingError) as e:
            print(f"Retrying: {e}")
            time.sleep(5)
            return self.request_OHLCV(interval=interval)
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
        for metric in ("timestamp", "open", "high", "low", "close", "volume"):
            self.data[metric] = [float(point[metric]) for point in OHLCV]
        print(f"{self}\tTotal size: {len(self.data['open'])}")

    def get_random_OHLCV_window(self, n_points=200):
        print(f"{self}: Getting random window")
        start_index = random.randint(0, len(self.data['open']) - n_points)
        end_index = start_index + n_points
        return {
            "timestamps": self.data['timestamps'][start_index:end_index],
            "open": self.data['open'][start_index:end_index],
            "high": self.data['high'][start_index:end_index],
            "low": self.data['low'][start_index:end_index],
            "close": self.data['close'][start_index:end_index],
            "volume": self.data['volume'][start_index:end_index],
        }


class InstrumentDataset:
    def __init__(self, min_points=50000, db_host="mongodb://localhost:27017/", api_key=API_KEY):
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

    def get_random_window(self, n_points=200, minute_timesteps=True, split='test', ta=None):
        instrument_id = random.choice(self.test_ids if split == 'test' else self.train_ids)
        instrument = KaikoInstrument(self.__instrument_collection.find_one({'_id': instrument_id}))
        window = instrument.get_random_OHLCV_window(n_points=n_points)
        if minute_timesteps:
            window['timestamps'] = [(t - window['timestamps'][0]) / 60000 for t in window['timestamps']]
        return window

    def get_instrument_ids(self):
        self.exchanges = self.__exchange_collection.find()
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

    def request_exchanges(self, override_cache=False):
        r = requests.get('https://reference-data-api.kaiko.io/v1/exchanges')
        self.exchanges = r.json()

    def request_OHLCVs(self, override_cache=False, interval='1m'):
        r = requests.get('https://reference-data-api.kaiko.io/v1/instruments')
        instruments = r.json()['data']
        for i, instrument_data in enumerate(instruments):
            instrument = KaikoInstrument(instrument_data, interval=interval)
            if self.__instrument_collection.find_one({'_id': str(instrument)}) is not None and not override_cache:
                print(f"{str(instrument)}: Already Cached")
                continue
            else:
                self.to_update.append(str(instrument))
                instrument.request_OHLCV()
            if i % 100 == 0:
                print(f"Saving {i}/{len(instruments)}")
                self.save()
        self.save()

    def save(self):
        print(f"To update: {self.to_update}")
        for instrument in self.to_update:
            self.__instrument_collection.update_one({'_id': str(instrument)}, {"$set": instrument.to_dict()}, upsert=True)
        if len(self.exchanges) > 0:
            self.__exchange_collection.insert_many(self.exchanges)
        self.to_update = []


class RandomMarketData:
    def __init__(self):
        self.data = None
        self.current_point = 0
        self.market_encoder = None

    def load_csv(self):
        pass

    def load_raw(self, data):
        self.data = data

    def get_last_n_points(self, n):
        data = []
        for i in range(n):
            index = self.current_point - n + i
            data.append(self.data[index] if index >= 0 else -1)
        return data

    def retrieve_data(self, start, end, interval):
        data = []
        for i in range(start, end, interval):
            data.append(self.data[i])
        return data

    def next(self):
        self.current_point += 1
        return self.data[self.current_point - 1]

    def reset(self):
        self.current_point = 0


if __name__ == "__main__":
    kd = InstrumentDataset(min_points=100000)
    start = time.time()
    print(kd.get_random_window(n_points=10))
    print(time.time() - start)
    # kd.request_OHLCVs()

    # if technical_indicators is not None:
    #     window['timestamps'] = window['timestamps']
    #     window = pandas.DataFrame.from_dict(window)
    #     print(window)
    #     all_ta = ta.add_all_ta_features(window, open="open", high="high", low="low", close="close", volume="volume")
    #     print(all_ta)
    #

