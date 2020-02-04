# AIO class for representation of market data and incremental access
import json
import pymongo
import random
import time
import requests
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
from requests.exceptions import HTTPError, ChunkedEncodingError

API_KEY = 'b4fda6c9cfa8c173209c61171bd1b4aa'


class KaikoInstrument:
    def __init__(self, instrument, interval='1m', api_key=API_KEY):
        self.api_key = api_key
        self.instrument = instrument
        self.interval = interval
        self.exchange = self.instrument["exchange_code"]
        self.class_name = self.instrument["class"]
        self.code = self.instrument["code"]
        self.timestamps = instrument["timestamps"] if "timestamps" in instrument else []
        self.open = instrument["open"] if "open" in instrument else []
        self.high = instrument["high"] if "high" in instrument else []
        self.low = instrument["low"] if "low" in instrument else []
        self.close = instrument["close"] if "close" in instrument else []
        self.volume = instrument["volume"] if "volume" in instrument else []
        self.unsaved = False

    def __repr__(self):
        return ':'.join([self.exchange, self.class_name, self.code, self.interval])

    def toDict(self):
        self.instrument.update({
            "_id": self.__repr__(),
            "interval": self.interval,
            "timestamps": self.timestamps,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        })
        return self.instrument

    def request_OHLCV(self, interval='1m'):
        self.unsaved = True
        try:
            initial_res = requests.get(
                f'https://eu.market-api.kaiko.io/v1/data/trades.latest/exchanges/{self.exchange}/{self.class_name}/{self.code}/aggregations/ohlcv',
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
        self.timestamps, self.open, self.high, self.low, self.close, self.volume = (
            [float(point[metric]) for point in OHLCV] for metric in ("timestamp", "open", "high", "low", "close", "volume")
        )

        print(f"{self}\tTotal size: {len(self.open)}")

    def get_random_OHLCV_window(self, n_points=200):
        print(f"{self}: Getting random window")
        assert len(self.open) > n_points, "Less than required points..."
        assert len(self.open) == len(self.close) == len(self.high) == len(self.low) == len(self.volume), "Same len"
        start_index = random.randint(0, len(self.open) - n_points)
        end_index = start_index + n_points
        return {
            "timestamps": self.timestamps[start_index:end_index],
            "open": self.open[start_index:end_index],
            "high": self.high[start_index:end_index],
            "low": self.low[start_index:end_index],
            "close": self.close[start_index:end_index],
            "volume": self.volume[start_index:end_index],
        }


class KaikoData:
    def __init__(self, db_host="mongodb://localhost:27017/", api_key=API_KEY):
        self.api_key = api_key
        self.exchanges = {}
        self.to_update = []
        self.instrument_ids = []
        self.__mongo_client = MongoClient(db_host)
        self.__db = self.__mongo_client["crypto_rl"]
        self.__instrument_collection = self.__db["instruments"]
        self.__exchange_collection = self.__db["exchanges"]
        # self.__instrument_collection.drop_index([('data_id', pymongo.ASCENDING)])

    def save(self, overwrite=False):
        print(f"To update: {self.to_update}")
        for instrument in self.to_update:
            self.__instrument_collection.update_one({'_id': str(instrument)}, {"$set": instrument.toDict()}, upsert=True)
        if len(self.exchanges) > 0:
            self.__exchange_collection.insert_many(self.exchanges)
        self.to_update = []

    def load(self, min_points=1000):
        self.exchanges = self.__exchange_collection.find()
        self.instrument_ids = self.__instrument_collection.distinct('_id', filter={f'open.{min_points}': {'$exists': True}})

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

    def get_random(self, n_points=200):
        instrument_id = random.choice(self.instrument_ids)
        instrument = KaikoInstrument(self.__instrument_collection.find_one({'_id': instrument_id}))
        return instrument, instrument.get_random_OHLCV_window(n_points=n_points)


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
    kd = KaikoData()
    kd.prune()
    # kd.request_OHLCVs()


