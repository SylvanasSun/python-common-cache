#!/usr/bin/env python
# -*- coding: utf-8 -*-

def fifo_for_evict(cache_dict, evict_number=1):
    """
    Use FIFO(First In First Out) strategy for evicting, it will find an item by earliest birthday date then remove it.

    Test:
    >>> from common_cache import CacheItem
    >>> import time
    >>> dict = {}
    >>> dict['a'] = CacheItem(key='a', value=0, expire=5)
    >>> time.sleep(1)
    >>> dict['b'] = CacheItem(key='b', value=1, expire=5)
    >>> time.sleep(1)
    >>> dict['c'] = CacheItem(key='c', value=2, expire=5)
    >>> len(dict)
    3
    >>> evicted_keys = fifo_for_evict(dict, evict_number=2)
    >>> len(dict)
    1
    >>> len(evicted_keys)
    2
    """
    ordered_dict = sorted(cache_dict.items(), key=lambda t: t[1]['birthday'])
    evicted_keys = []
    for i in range(evict_number):
        item = ordered_dict[i]
        key = item[0]
        cache_dict.pop(key)
        evicted_keys.append(key)
    return evicted_keys


def lru_for_evict(cache_dict, evict_number=1):
    """
    Use LRU(Least Recently Used) strategy for evicting, the item that last used of time is the earliest will be removed.

    Test:
    >>> from common_cache import CacheItem
    >>> import time
    >>> dict = {}
    >>> dict['a'] = CacheItem(key='a', value=0, expire=5)
    >>> time.sleep(1)
    >>> dict['b'] = CacheItem(key='b', value=1, expire=5)
    >>> time.sleep(1)
    >>> dict['c'] = CacheItem(key='c', value=2, expire=5)
    >>> len(dict)
    3
    >>> evicted_keys = lru_for_evict(dict, evict_number=2)
    >>> len(dict)
    1
    >>> len(evicted_keys)
    2
    """
    ordered_dict = sorted(cache_dict.items(), key=lambda t: t[1]['last_used_time'])
    evicted_keys = []
    for i in range(evict_number):
        item = ordered_dict[i]
        key = item[0]
        cache_dict.pop(key)
        evicted_keys.append(key)
    return evicted_keys


def lfu_for_evict(cache_dict, evict_number=1):
    """
    Use LFU(Least Frequently Used) strategy for evicting, the item that number of hits is the least will be removed.

    Test:
    >>> from common_cache import CacheItem
    >>> import time
    >>> dict = {}
    >>> dict['a'] = CacheItem(key='a', value=0, expire=5)
    >>> time.sleep(1)
    >>> dict['b'] = CacheItem(key='b', value=1, expire=5)
    >>> time.sleep(1)
    >>> dict['c'] = CacheItem(key='c', value=2, expire=5)
    >>> len(dict)
    3
    >>> evicted_keys = lfu_for_evict(dict, evict_number=1)
    >>> len(dict)
    2
    >>> len(evicted_keys)
    1
    """
    ordered_dict = sorted(cache_dict.items(), key=lambda t: t[1]['hit_counts'])
    evicted_keys = []
    for i in range(evict_number):
        item = ordered_dict[i]
        key = item[0]
        cache_dict.pop(key)
        evicted_keys.append(key)
    return evicted_keys


class EvictionStrategy(object):
    FIFO = fifo_for_evict
    LRU = lru_for_evict
    LFU = lfu_for_evict


if __name__ == '__main__':
    import doctest

    doctest.testmod()
