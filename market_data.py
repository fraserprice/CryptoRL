# AIO class for representation of market data and incremental access
import json
import os
import random
import time

import requests
from tqdm import tqdm
from datetime import datetime

API_KEY = 'b4fda6c9cfa8c173209c61171bd1b4aa'


class KaikoInstrument:
    def __init__(self, instrument, interval='1m', ohlcv=None, request=False, api_key=API_KEY):
        self.api_key = api_key
        self.instrument = instrument
        self.interval = interval
        self.exchange = self.instrument["exchange_code"]
        self.class_name = self.instrument["class"]
        self.code = self.instrument["code"]
        self.OHLCV = ohlcv
        if self.OHLCV is None and request:
            self.request_OHLCV(self.interval)

    def __repr__(self):
        return ':'.join([self.exchange, self.class_name, self.code, self.interval])

    def toDict(self):
        self.instrument.update({
            "id": self.__repr__(),
            "interval": self.interval,
            "ohlcv": self.OHLCV
        })
        return self.instrument

    def request_OHLCV(self, interval='1m'):
        initial_res = requests.get(
            f'https://eu.market-api.kaiko.io/v1/data/trades.latest/exchanges/{self.exchange}/{self.class_name}/{self.code}/aggregations/ohlcv',
            headers={f'X-Api-Key': self.api_key, 'Accept': 'application/json'},
            params={'interval': interval, 'page_size': 100000}
        )
        if initial_res.status_code != 200:
            print("Retrying...")
            time.sleep(5)
            return self.request_OHLCV(interval=interval)
        self.OHLCV = initial_res.json()['data']
        if 'next_url' in initial_res.json():
            next_url = initial_res.json()['next_url']
            while True:
                next_res = requests.get(next_url,
                                        headers={f'X-Api-Key': self.api_key, 'Accept': 'application/json', },
                                        params={'interval': '1m'})
                page_data = next_res.json()['data']
                self.OHLCV += page_data
                if 'next_url' in next_res.json():
                    print(next_res.json()['next_url'])
                    next_url = next_res.json()['next_url']
                    continue
                break
        for ohlcv in self.OHLCV:
            for key in ohlcv.keys():
                ohlcv[key] = float(ohlcv[key])
        print(f"{self}\tRequested 1m interval. Total sizoe: {len(self.OHLCV)}")

    def get_random_OHLCV_window(self, n_points=200):
        assert len(self.OHLCV) > n_points, "Less than required points..."
        start_index = random.randint(0, len(self.OHLCV) - n_points)
        return self.OHLCV[start_index:start_index + n_points]


class KaikoData:
    def __init__(self, filepath, api_key=API_KEY):
        self.api_key = api_key
        self.exchanges = {}
        self.instruments = {}
        assert filepath[-5:].lower() == '.json', "Must be JSON file"
        self.filepath = filepath
        if os.path.isfile(self.filepath):
            self.load()

    def save(self):
        info = {"exchanges": self.exchanges,
                "instruments": {inst_id: instrument.toDict() for inst_id, instrument in self.instruments.items()}}
        with open(self.filepath, 'w') as fp:
            json.dump(info, fp, indent=4)

    def load(self):
        with open(self.filepath, 'r') as data:
            data = json.load(data)
            self.exchanges = data["exchanges"]
            self.instruments = {
                str(ki): ki
                for ki in (KaikoInstrument(info, ohlcv=info["ohlcv"], interval=info["interval"]) for id_, info in data["instruments"].items())
            }

    def request_exchanges(self, override_cache=False):
        r = requests.get('https://reference-data-api.kaiko.io/v1/exchanges')
        self.exchanges = r.json()

    def request_OHLCVs(self, override_cache=False, interval='1m'):
        r = requests.get('https://reference-data-api.kaiko.io/v1/instruments')
        instruments = r.json()['data']
        for i, instrument_data in enumerate(instruments):
            instrument = KaikoInstrument(instrument_data, interval=interval, request=False)
            if str(instrument) in self.instruments and not override_cache:
                continue
            else:
                instrument.request_OHLCV()
                self.instruments[str(instrument)] = instrument
            if i % 5 == 0:
                print(f"Saving {i}/{len(instruments)}")
                self.save()
        self.save()

    def prune(self, min_points=1000):
        rem = set()
        for id_, instrument in self.instruments.items():
            if len(instrument.OHLCV) < min_points:
                rem.add(id_)
        for id_ in rem:
            del self.instruments[id_]
        self.save()

    def get_random(self, n_points=200):
        instrument = random.choice(self.instruments.values())
        return instrument, instrument.get_random_OHLCV_window(n_points=n_points)




class MarketData:
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
    kd = KaikoData("data/data.json")
    kd.request_OHLCVs()


