#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import threading
import time
import weakref


def basic_cleanup(self):
    """
    Test:

    >>> from common_cache import Cache
    >>> import time
    >>> cache = Cache(expire=1, cleanup_func=basic_cleanup, regularly_cleanup=False)
    >>> cache.put('a', value=0)
    >>> cache.put('b', value=1)
    >>> cache.put('c', value=2)
    >>> cache.put('d', value=3, expire=3)
    >>> cache.size()
    4
    >>> time.sleep(1)
    >>> cache.put('e', 4)
    >>> cache.get('a') == None
    True
    >>> cache.get('d') == None
    False
    """
    next_expire = None
    keys_to_delete = []
    if self.expire is not None and self.expire > 0:
        # cleanup invalid cache item until the meet valid cache item and record next expire time
        for k, item in self.cache_items.items():
            if item.is_dead():
                keys_to_delete.append(k)
            else:
                next_expire = item.remaining_survival_time()
                break

    # if direct delete will to cause an error: OrderedDict mutated during iteration
    # so use delay delete
    for k in keys_to_delete:
        self.cache_items.pop(k)

    # if reach the upper limit of capacity then will execute evict by eviction strategy
    while (len(self.cache_items) > self.capacity):
        evicted_keys = self.evict_func(cache_dict=self.cache_items, evict_number=self.evict_number)
        self.logger.debug('Evict operation is completed, count: %s, keys: %s' % (len(evicted_keys), evicted_keys))

    return next_expire


class CleanupSupervisorThread(threading.Thread):
    """
    This is a daemon thread for regularly cleanup invalid cache.
    """

    def __init__(self, cache, logger, interval=60, name='CacheCleanupSupervisor', daemon=True):
        self.cache_ref = weakref.ref(cache)
        self.interval = interval
        self.logger = logger
        self.flag = True
        super(CleanupSupervisorThread, self).__init__(name=name, daemon=daemon)

    def stop(self):
        self.flag = False

    def run(self):
        self.logger.debug('Thread[%s] start running and will to work in the after 10 seconds' % self.name)
        time.sleep(10)

        while self.cache_ref() and self.flag:
            cache = self.cache_ref()
            if cache:
                next_expire = cache.cleanup(cache)
                if next_expire is None:
                    next_expire = self.interval
                else:
                    next_expire = next_expire + 1
                self.logger.debug('Cleanup is completed and %s go to sleep, next cleanup date: %s'
                                  % (self.name, datetime.datetime.fromtimestamp(time.time() + next_expire)))
                time.sleep(next_expire)
            cache = None


if __name__ == '__main__':
    import doctest

    doctest.testmod()
