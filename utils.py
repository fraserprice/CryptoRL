from collections import defaultdict


def get_minute_difference(unix_start, unix_end):
    divisor = 1000000 if unix_start > 1e13 else 1000

    return (unix_end - unix_start) / (60 * divisor)


# noinspection PyTypeChecker
def aggregate_ohlcv_window(window, aggregate=5, minute_timestamps=True):
    n_points = len(window['open'])
    new_window_len = n_points // aggregate
    agg_window = defaultdict(list)
    for i in range(0, n_points, aggregate):
        agg_window['timestamps'].append(window['timestamps'][i + aggregate - 1])
        agg_window['open'].append(window['open'][i])
        agg_window['close'].append(window['close'][i + aggregate - 1])
        agg_window['high'].append(max(window['high'][i:i + aggregate]))
        agg_window['low'].append(min(window['low'][i:i + aggregate]))
        agg_window['volume'].append(sum(window['volume'][i:i + aggregate]))
    if minute_timestamps:
        agg_window['timestamps'] = list(reversed([get_minute_difference(agg_window['timestamps'][0], t) for t in agg_window['timestamps']]))
    assert sum([len(l) for l in agg_window.values()]) == new_window_len * len(agg_window)
    return agg_window


def get_dict_of_lists_slice(dict_of_lists, start=None, end=None, keys=None):
    if keys is not None:
        dict_of_lists = {k: v for k, v in dict_of_lists.items() if k in keys}
    if start is None:
        assert end is not None
        return {k: l[:end] for k, l in dict_of_lists.items()}
    elif end is None:
        assert start is not None
        return {k: l[start:] for k, l in dict_of_lists.items()}
    else:
        return {k: l[start:end] for k, l in dict_of_lists.items()}

