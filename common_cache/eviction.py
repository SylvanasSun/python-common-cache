#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections


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
    The parameter cache_dict must is an OrderedDict because its implementation based on the OrderedDict and reinsert key
    and value when every time to get the cache, this operation will make the cache of the often used is in the
    tail of OrderedDict and head of the OrderedDict is a cache of the least recently used.

    Test:
    >>> import collections
    >>> from common_cache import CacheItem
    >>> dict = {'a' : 0, 'b' : 1}
    >>> lru_for_evict(dict)
    Traceback (most recent call last):
        ...
    ValueError: Not supported type <class 'dict'>
    >>> cache = collections.OrderedDict()
    >>> cache['a'] = CacheItem(key='a', value=0, expire=3)
    >>> cache['b'] = CacheItem(key='b', value=1, expire=3)
    >>> cache['c'] = CacheItem(key='c', value=2, expire=3)
    >>> cache['d'] = CacheItem(key='d', value=3, expire=3)
    >>> lru_for_evict(cache)
    ['a']
    >>> len(cache)
    3
    >>> lru_for_evict(cache, evict_number=2)
    ['b', 'c']
    >>> len(cache)
    1
    """
    if not isinstance(cache_dict, collections.OrderedDict):
        raise ValueError('Not supported type %s' % type(cache_dict))
    evicted_keys = []
    for i in range(evict_number):
        item = cache_dict.popitem(last=False)
        evicted_keys.append(item[0])
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
