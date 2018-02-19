#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time


class Cache(object):
    def __init__(self): pass


class CacheItem(dict):
    """
    The class represents an item for the cache, it is a subclass that extends dictionary
    and it will record some info for example key, value, expire, hit counts...

    Test:

    >>> import time
    >>> key = 'name'
    >>> value = 'SylvanasSun'
    >>> now = time.time()
    >>> item = CacheItem(key=key, value=value, expire=2)
    >>> now == item.birthday()
    True
    >>> now == item.last_used_time()
    True
    >>> item[key]
    'SylvanasSun'
    >>> item.is_dead()
    False
    >>> item['hit_counts']
    0
    >>> time.sleep(2)
    >>> item.is_dead()
    True
    >>> item.remaining_survival_time()
    0
    >>> now = time.time()
    >>> item.update_last_used_time()
    >>> now == item.last_used_time()
    True
    >>> item.refresh_expire(5)
    >>> item.is_dead()
    False
    >>> item.update_hit_count()
    >>> item.compute_hit_rate(total_visits=10)
    0.1
    >>> item.clear()
    >>> item
    {}
    """

    def __init__(self, key, value, expire, hit_counts=0, *args, **kwargs):
        super(CacheItem, self).__init__(*args, **kwargs)
        self.itemlist = list(super(CacheItem, self).keys())
        self.__setitem__(key, value)
        timestamp = time.time()
        self.__setitem__('birthday', timestamp)
        self.__setitem__('last_used_time', timestamp)
        self.__setitem__('expire', timestamp + expire)
        self.__setitem__('hit_counts', hit_counts)

    def is_dead(self):
        if self.__getitem__('expire') <= time.time():
            return True
        else:
            return False

    def refresh_expire(self, expire):
        self.__setitem__('expire', time.time() + expire)

    def birthday(self):
        return self.__getitem__('birthday')

    def last_used_time(self):
        return self.__getitem__('last_used_time')

    def update_last_used_time(self):
        self.__setitem__('last_used_time', time.time())

    def remaining_survival_time(self):
        expire = self.__getitem__('expire')
        now = time.time()
        remain = expire - now
        if remain < 0:
            return 0
        else:
            return remain

    def update_hit_count(self):
        self.__setitem__('hit_counts', self.__getitem__('hit_counts') + 1)

    def compute_hit_rate(self, total_visits):
        return self.__getitem__('hit_counts') / total_visits

    def __setitem__(self, key, value):
        self.itemlist.append(key)
        super(CacheItem, self).__setitem__(key, value)

    def __iter__(self):
        return iter(self.itemlist)

    def keys(self):
        return self.itemlist

    def values(self):
        return [self[key] for key in self]

    def itervalues(self):
        return (self[key] for key in self)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
