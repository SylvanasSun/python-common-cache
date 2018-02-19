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
    >>> item = CacheItem(key=key, value=value, expire=5)
    >>> item[key]
    'SylvanasSun'
    >>> item.is_dead()
    False
    >>> item['hit_counts']
    0
    >>> time.sleep(5)
    >>> item.is_dead()
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
        self.__setitem__('expire', time.time() + expire)
        self.__setitem__('hit_counts', hit_counts)

    def is_dead(self):
        if self.__getitem__('expire') <= time.time():
            return True
        else:
            return False

    def refresh_expire(self, expire):
        self.__setitem__('expire', time.time() + expire)

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
