#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import datetime
import functools
import inspect
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from common_cache.cleanup import CleanupSupervisorThread, basic_cleanup
from common_cache.eviction import EvictionStrategy


def _enable_lock(func):
    """
    The decorator for ensuring thread-safe when current cache instance is concurrent status.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.is_concurrent:
            with self._rlock:
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


def _enable_cleanup(func):
    """
    Execute cleanup operation when the decorated function completed.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        result = func(*args, **kwargs)
        self.cleanup(self)
        return result

    return wrapper


def _enable_thread_pool(func):
    """
    Use thread pool for executing a task if self.enable_thread_pool is True.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if self.enable_thread_pool and hasattr(self, 'thread_pool'):
            future = self.thread_pool.submit(func, *args, **kwargs)
            timeout = kwargs.get('timeout')
            if timeout is None:
                timeout = 2
            try:
                result = future.result(timeout=timeout)
            except TimeoutError as e:
                self.logger.exception(e)
                result = None
            return result
        else:
            return func(*args, **kwargs)

    return wrapper


def _check_function_obj(param_length):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            func_obj = args[1]
            try:
                func_name = func_obj.__name__
            except AttributeError:
                func_name = 'None'

            if not inspect.isfunction(func_obj):
                self.logger.warning('Parameter %s must be a function object' % func_name)
                return False

            if len(inspect.signature(func_obj).parameters) != param_length:
                self.logger.warning('Parameter %s must be have %s parameters' % (func_name, param_length))
                return False

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _init_thread_pool(max_workers=10, thread_name_prefix='COMMON-CACHE-'):
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)


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
        self.key = key
        timestamp = time.time()
        self.__setitem__('birthday', timestamp)
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

    def statistic_record(self, total_access_count):
        hit_counts = self.__getitem__('hit_counts')
        return {
            'key': self.key,
            'value': self.__getitem__(self.key),
            'hit_counts': hit_counts,
            'miss_counts': total_access_count - hit_counts,
            'hit_percentage': self.compute_hit_rate(total_access_count),
            'birthday': str(datetime.datetime.fromtimestamp(self.__getitem__('birthday'))),
            'expire': str(datetime.datetime.fromtimestamp(self.__getitem__('expire'))),
            'remaining_survival_time': self.remaining_survival_time()
        }

    def __setitem__(self, key, value):
        self.itemlist.append(key)
        super(CacheItem, self).__setitem__(key, value)

    def __iter__(self):
        return iter(self.itemlist)

    def __eq__(self, other):
        if isinstance(other, CacheItem):
            return dict.__eq__(self, other) and all(map(lambda a, b: a == b, self, other))
        return dict.__eq__(self, other)

    def keys(self):
        return self.itemlist

    def values(self):
        return [self[key] for key in self]

    def itervalues(self):
        return (self[key] for key in self)


class Cache(object):
    """
    The class Cache represents an application level cache in the memory and each its object
    are all an individual cache instance, it supports many configuration items for you dynamic
    custom your cache such as thread pool, cleanup function, regularly cleanup strategy, read after refresh expire,
    evict function(supports three kinds of strategy: FIFO, LRU, LFU, you can also use your own function) and so on.

    It supports to build a multi-level cache system by cache_loader(key) and cache_writer(key, value),
    the former will can load cache from other cache system or data source when the cache is miss and
    the latter will can write the results from the data source to another cache system or data source.

    If you only want a method is simple enough for use cache so you can also use default configuration
    and directly create an object of Cache, example: cache = Cache(), about default configuration:
    the cache will use LRU(Least Recently Used) strategy for evicting when the capacity is full and start a
    daemon thread for regularly cleanup invalid cache item and refresh expire time when after read a cache
    item and init a thread pool for handle some operations such as get, put and pop and so on.

    Notice: cache will invoke cleanup() for cleanup invalid cache item when after each use of get()/put()/pop().

    The usage is very simple you can easily implement caching with a decorator
    @cache_instance.access_cache(), look at following example code:

    >>> cache = Cache() # default configuration
    >>> @cache.access_cache(key='foo')
    ... def foo():
    ...     return 'data from data source'
    >>> foo() # cache does not have data so data from a data source
    'data from data source'
    >>> cache.get(key='foo') # default enable auto update cache
    'data from data source'
    >>> cache.put(key='foo', value='data from cache') # update cache
    >>> foo() # return data from cache
    'data from cache'
    """

    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
    cache_instance_id_counter = 0

    def __init__(self, capacity=1024, expire=60, evict_func=EvictionStrategy.LRU,
                 evict_number=1, instance_name='CACHE-INSTANCE', cleanup_func=basic_cleanup,
                 regularly_cleanup=True, regularly_cleanup_interval=60, is_concurrent=True,
                 read_after_refresh_expire=True, log_filename=None, log_format=None, log_level=logging.INFO,
                 enable_thread_pool=True, thread_pool_init_func=_init_thread_pool, cache_loader=None):
        self.capacity = capacity
        self.expire = expire
        self.evict_func = evict_func
        self.cleanup = cleanup_func
        self.evict_number = evict_number
        self.total_access_count = 0
        self.read_after_refresh_expire = read_after_refresh_expire
        self.enable_thread_pool = enable_thread_pool
        self.cache_items = collections.OrderedDict()
        if is_concurrent:
            self._rlock = threading.RLock()
        self.is_concurrent = is_concurrent

        instance_name = self._generate_cache_instance_name(basic_name=instance_name)
        self._init_logger(level=log_level, filename=log_filename, format=log_format, name=instance_name)

        self.regularly_cleanup_interval = regularly_cleanup_interval
        if regularly_cleanup:
            cleanup_supervisor = CleanupSupervisorThread(cache=self, logger=self.logger,
                                                         interval=regularly_cleanup_interval)
            cleanup_supervisor.start()
            self.cleanup_supervisor = cleanup_supervisor

        if enable_thread_pool and thread_pool_init_func is not None:
            self.thread_pool = thread_pool_init_func()
            self.logger.debug('Initialize thread pool completed %s' % self.thread_pool)

        self.cache_loader = cache_loader
        if self.cache_loader is not None:
            cache_loader_name = self.cache_loader.__name__
        else:
            cache_loader_name = 'None'

        log_message = 'Cache initialize is completed \n'
        log_message += '[ capacity: %s, expire: %s, eviction strategy: %s, evict number: %s, instance name: %s \n'
        log_message += 'cleanup function: %s, regularly cleanup: %s, regularly cleanup interval: %s, is concurrent: %s \n'
        log_message += 'read after refresh expire: %s, log filename: %s, log format: %s, log level: %s \n'
        log_message += 'enable thread pool: %s, thread pool init function: %s, cache loader: %s ]'
        self.logger.debug(log_message
                          % (capacity, expire, evict_func.__name__, evict_number, instance_name,
                             cleanup_func.__name__, regularly_cleanup, regularly_cleanup_interval, is_concurrent,
                             read_after_refresh_expire, log_filename, log_format, log_level,
                             enable_thread_pool, thread_pool_init_func.__name__, cache_loader_name))

    @_enable_lock
    def size(self):
        """
        >>> cache = Cache()
        >>> cache.put('a', 0)
        >>> cache.put('b', 1)
        >>> cache.put('c', 2)
        >>> cache.size()
        3
        >>> cache.pop('c')
        2
        >>> cache.size()
        2
        """
        return len(self.cache_items)

    @_enable_lock
    def clear(self):
        """
        >>> cache = Cache()
        >>> cache.put('a', 0)
        >>> cache.put('b', 1)
        >>> cache.size()
        2
        >>> cache.clear()
        >>> cache.size()
        0
        """
        self.cache_items.clear()
        self.total_access_count = 0
        self.logger.debug('Cache clear operation is completed')

    @_check_function_obj(param_length=2)
    @_enable_lock
    def replace_evict_func(self, func):
        """
        >>> cache = Cache()
        >>> def evict(dict, evict_number=10): pass
        >>> cache.replace_evict_func(evict)
        True
        >>> def evict_b(dict): pass
        >>> cache.replace_evict_func(evict_b)
        Parameter evict_b must be have 2 parameters
        False
        >>> def evict_c(dict, a, b): pass
        >>> cache.replace_evict_func(evict_c)
        Parameter evict_c must be have 2 parameters
        False
        """
        self.evict_func = func
        return True

    @_check_function_obj(param_length=1)
    @_enable_lock
    def replace_cleanup_func(self, func):
        """
        >>> cache = Cache()
        >>> def cleanup(self): pass
        >>> cache.replace_cleanup_func(cleanup)
        True
        >>> def cleanup_b(self, other): pass
        >>> cache.replace_cleanup_func(cleanup_b)
        Parameter cleanup_b must be have 1 parameters
        False
        """
        self.cleanup = func
        return True

    @_check_function_obj(param_length=1)
    @_enable_lock
    def with_cache_loader(self, func):
        """
        >>> cache = Cache()
        >>> def cache_loader(key): pass
        >>> cache.with_cache_loader(cache_loader)
        True
        >>> def cache_loader_b(key, value): pass
        >>> cache.with_cache_loader(cache_loader_b)
        Parameter cache_loader_b must be have 1 parameters
        False
        """
        self.cache_loader = func
        return True

    @_enable_lock
    def stop_regularly_cleanup(self):
        """
        >>> cache = Cache()
        >>> cache.stop_regularly_cleanup()
        True
        >>> cache.stop_regularly_cleanup()
        Current not have a regularly cleanup thread is existent
        False
        """
        if hasattr(self, 'cleanup_supervisor') and self.cleanup_supervisor is not None:
            self.cleanup_supervisor.stop()
            self.logger.debug('Regularly cleanup thread %s is closed' % self.cleanup_supervisor.name)
            self.cleanup_supervisor = None
            return True
        else:
            self.logger.warning('Current not have a regularly cleanup thread is existent')
            return False

    @_enable_lock
    def start_regularly_cleanup(self, regularly_cleanup_interval=None):
        """
        >>> cache = Cache(regularly_cleanup=False, log_level=logging.ERROR)
        >>> cache.start_regularly_cleanup()
        True
        >>> cache.start_regularly_cleanup()
        False
        """
        if not hasattr(self, 'cleanup_supervisor') or self.cleanup_supervisor is None:
            if regularly_cleanup_interval is None or not isinstance(regularly_cleanup_interval, int):
                regularly_cleanup_interval = self.regularly_cleanup_interval

            cleanup_supervisor = CleanupSupervisorThread(cache=self, logger=self.logger,
                                                         interval=regularly_cleanup_interval)
            cleanup_supervisor.start()
            self.cleanup_supervisor = cleanup_supervisor
            return True
        else:
            self.logger.warning('Already have a regularly cleanup thread - %s' % self.cleanup_supervisor.name)
            return False

    @_enable_lock
    def with_thread_pool(self, init_func=None):
        """
        >>> cache = Cache(enable_thread_pool=False, log_level=logging.ERROR)
        >>> cache.with_thread_pool()
        True
        >>> cache.with_thread_pool()
        False
        """
        if not hasattr(self, 'thread_pool') or self.thread_pool is None:
            if init_func is not None:
                self.thread_pool = init_func()
            else:
                self.thread_pool = _init_thread_pool()
            self.enable_thread_pool = True
            self.logger.debug('Initialize thread pool completed %s' % self.thread_pool)
            return True
        else:
            self.logger.warning('Already have a thread pool instance - %s' % self.thread_pool)
            return False

    @_enable_lock
    def unable_thread_pool(self):
        """
        >>> cache = Cache()
        >>> cache.unable_thread_pool()
        True
        >>> cache.unable_thread_pool()
        Current not have a thread pool is existent
        False
        """
        if self.enable_thread_pool and hasattr(self, 'thread_pool') and self.thread_pool is not None:
            self.thread_pool.shutdown()
            self.thread_pool = None
            self.enable_thread_pool = False
            self.logger.debug('Remove thread pool completed')
            return True
        else:
            self.logger.warning('Current not have a thread pool is existent')
            return False

    @_enable_lock
    def shutdown_thread_pool(self):
        """
        >>> cache = Cache()
        >>> cache.shutdown_thread_pool()
        True
        >>> cache.unable_thread_pool()
        True
        >>> cache.shutdown_thread_pool()
        Current not have a thread pool is existent
        False
        """
        if self.enable_thread_pool and hasattr(self, 'thread_pool') and self.thread_pool is not None:
            self.thread_pool.shutdown()
            return True
        else:
            self.logger.warning('Current not have a thread pool is existent')
            return False

    @_enable_lock
    def set_capacity(self, new_capacity):
        """
        >>> cache = Cache()
        >>> cache.set_capacity(100)
        >>> cache.capacity
        100
        >>> cache.set_capacity('haha')
        Parameter new_capacity haha must be greater than 0 and is an integer
        >>> cache.capacity
        100
        """
        if not isinstance(new_capacity, int) or new_capacity <= 0:
            self.logger.warning('Parameter new_capacity %s must be greater than 0 and is an integer' % new_capacity)
            return
        self.capacity = new_capacity

    @_enable_lock
    def set_expire(self, new_expire):
        """
        >>> cache = Cache()
        >>> cache.set_expire(40)
        >>> cache.expire
        40
        >>> cache.set_expire('haha')
        Parameter new_expire haha must be positive number
        >>> cache.expire
        40
        """
        if not isinstance(new_expire, int) or new_expire < 0:
            self.logger.warning('Parameter new_expire %s must be positive number' % new_expire)
            return
        self.expire = new_expire

    @_enable_lock
    def set_evict_number(self, new_evict_number):
        """
        >>> cache = Cache()
        >>> cache.set_evict_number(10)
        >>> cache.evict_number
        10
        >>> cache.set_evict_number('haha')
        Parameter new_evict_number haha must be greater than 0 and is an integer
        >>> cache.evict_number
        10
        """
        if not isinstance(new_evict_number, int) or new_evict_number <= 0:
            self.logger.warning(
                'Parameter new_evict_number %s must be greater than 0 and is an integer' % new_evict_number)
            return
        self.evict_number = new_evict_number

    @_enable_lock
    def with_read_after_refresh_expire(self, flag):
        """
        >>> cache = Cache()
        >>> cache.read_after_refresh_expire
        True
        >>> cache.with_read_after_refresh_expire(False)
        >>> cache.read_after_refresh_expire
        False
        >>> cache.with_read_after_refresh_expire('haha')
        Parameter flag haha must be boolean
        >>> cache.read_after_refresh_expire
        False
        """
        if not isinstance(flag, bool):
            self.logger.warning('Parameter flag %s must be boolean' % flag)
            return
        self.read_after_refresh_expire = flag

    @_enable_thread_pool
    @_enable_lock
    def statistic_record(self, desc=True, timeout=3, *keys):
        """
        Returns a list that each element is a dictionary of the statistic info of the cache item.
        """
        if len(keys) == 0:
            records = self._generate_statistic_records()
        else:
            records = self._generate_statistic_records_by_keys(keys)
        return sorted(records, key=lambda t: t['hit_counts'], reverse=desc)

    def _generate_statistic_records(self):
        records = []
        for item in self.cache_items.values():
            records.append(item.statistic_record(self.total_access_count))
        return records

    def _generate_statistic_records_by_keys(self, keys):
        records = []
        for key, item in self.cache_items.items():
            if key in keys:
                records.append(item.statistic_record(self.total_access_count))
        return records

    def access_cache(self, key=None, key_location_on_param=0, expire=None, auto_update=True,
                     cache_loader=None, cache_writer=None, timeout=1):
        """
        The decorator for simplifying of use cache, it supports auto-update
        cache(if parameter auto_update is True), load cache from other level cache
        system or data source and writes back the update result to the
        other level cache system or data source if cache miss.

        The parameter key assigns a key for access cache or update cache and
        if it is None so select a parameter as a key from the decorated function
        by key_location_on_param, notice: key and key_location_on_param cannot all is None.

        For function cache_loader() must is a one-parameter function and the parameter
        represent a key of the cache, if this parameter is None so use self.cache_loader(),
        if they all are None so not load cache from other caches system.

        For function cache_writer() must is a two-parameter function and the first parameter
        representing a key of the cache and the second parameter representing a value of the cache,
        notice: if the parameter auto_update is False so it will not execute.

        >>> import time
        >>> cache = Cache()
        >>> @cache.access_cache(key='a')
        ... def a():
        ...     return 'a from data source'
        >>> a()
        'a from data source'
        >>> cache.get('a')
        'a from data source'
        >>> cache.put(key='b', value='b from cache')
        >>> @cache.access_cache(key='b')
        ... def b():
        ...     return 'b from data source'
        >>> b()
        'b from cache'
        >>> c_key = 'c'
        >>> @cache.access_cache(key_location_on_param=0)
        ... def c(key):
        ...     return 'c from data source'
        >>> c(c_key)
        'c from data source'
        >>> cache.get(c_key)
        'c from data source'
        >>> @cache.access_cache(key='d', auto_update=False)
        ... def d():
        ...     return 'd from data source'
        >>> d()
        'd from data source'
        >>> cache.get('d') == None
        True
        >>> @cache.access_cache(key='e', cache_loader=lambda k: '%s from cache loader' % k)
        ... def e():
        ...     return 'e from data source'
        >>> e()
        'e from cache loader'
        >>> out_dict = {}
        >>> def writer(k, v):
        ...     out_dict[k] = v
        >>> @cache.access_cache(key='f', cache_writer=writer)
        ... def f():
        ...     return 'f from data source'
        >>> f()
        'f from data source'
        >>> time.sleep(1) # wait to execute complete because it in the other thread
        >>> out_dict
        {'f': 'f from data source'}
        >>> cache.with_cache_loader(lambda k: '%s from cache loader(global)' % k)
        True
        >>> @cache.access_cache(key='g')
        ... def g():
        ...     return 'g from data source'
        >>> g()
        'g from cache loader(global)'
        """

        def decorate(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                k = None
                if len(args) - 1 >= key_location_on_param:
                    k = args[key_location_on_param]
                if key is not None:
                    k = key
                cache_result = self.get(key=k, timeout=timeout)
                # if the cache is miss and cache loader is the existent
                # then query cache from cache loader
                if cache_result is None:
                    if cache_loader is None and self.cache_loader is not None:
                        cache_result = self.cache_loader(k)
                    elif cache_loader is not None:
                        cache_result = cache_loader(k)
                # if still miss then execute a function that is decorated
                # then update cache on the basis of parameter auto_update
                if cache_result is not None:
                    return cache_result
                else:
                    result = func(*args, **kwargs)
                if auto_update:
                    self.put(key=k, value=result, expire=expire, timeout=timeout)
                    if cache_writer is not None:
                        self.thread_pool.submit(cache_writer, k, result)
                return result

            return wrapper

        return decorate

    @_enable_thread_pool
    @_enable_lock
    @_enable_cleanup
    def get(self, key, timeout=1):
        """
        Test:

        >>> import time
        >>> cache = Cache(expire=2)
        >>> cache.put(key='a', value=0)
        >>> cache.put(key='b', value=1)
        >>> cache.put(key='c',value= 2)
        >>> cache.get('a')
        0
        >>> cache.get('b')
        1
        >>> cache.get('c')
        2
        >>> cache.get('e') == None
        True
        >>> time.sleep(2)
        >>> cache.put(key='e', value=4)
        >>> cache.get('a') == None
        True
        >>> cache.get('b') == None
        True
        >>> cache.get('c') == None
        True
        """
        if key not in self.cache_items:
            self.logger.debug('Cache item <%s> missing' % key)
            return None

        item = self.cache_items.pop(key)
        item.update_hit_count()
        if self.read_after_refresh_expire:
            item.refresh_expire(self.expire)
        value = item[key]
        self.cache_items[key] = item
        self.total_access_count += 1
        return value

    @_enable_thread_pool
    @_enable_lock
    @_enable_cleanup
    def put(self, key, value, expire=None, timeout=1):
        """
        Test:

        >>> cache = Cache(expire=2)
        >>> cache.put('a', 0)
        >>> cache.get('a')
        0
        >>> cache.put('a', 1)
        >>> cache.get('a')
        1
        """
        if key in self.cache_items:
            self.cache_items.pop(key)
        if expire is None:
            expire = self.expire
        self.cache_items[key] = CacheItem(key=key, value=value, expire=expire)

    @_enable_thread_pool
    @_enable_lock
    @_enable_cleanup
    def pop(self, key, timeout=1):
        """
        Test:

        >>> cache = Cache()
        >>> cache.put('a', 0)
        >>> cache.pop('a')
        0
        >>> cache.pop('b') == None
        True
        """
        if key not in self.cache_items:
            return None
        return self.cache_items.pop(key)[key]

    def __getitem__(self, key):
        """
        >>> cache = Cache()
        >>> cache.put('a', 0)
        >>> cache['a']
        0
        >>> cache.put('b', 1)
        >>> cache['b']
        1
        >>> cache['c'] == None
        True
        """
        return self.get(key)

    def __setitem__(self, key, value):
        """
        >>> cache = Cache()
        >>> cache['a'] = 0
        >>> cache['b'] = 1
        >>> cache.get('a')
        0
        >>> cache.get('b')
        1
        """
        return self.put(key, value)

    def __delitem__(self, key):
        """
        >>> cache = Cache()
        >>> cache.put('a', 0)
        >>> cache.put('b', 1)
        >>> del cache['a']
        >>> cache.get('a') == None
        True
        >>> cache.get('b')
        1
        >>> del cache['b']
        >>> cache.get('b') == None
        True
        """
        return self.pop(key)

    @_enable_lock
    def __contains__(self, key):
        return self.cache_items.__contains__(key)

    @_enable_lock
    def __iter__(self):
        return self.cache_items.__iter__()

    @_enable_lock
    def __reversed__(self):
        return self.cache_items.__reversed__()

    @_enable_lock
    def __sizeof__(self):
        return self.cache_items.__sizeof__()

    @_enable_lock
    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self.cache_items.items()))

    @_enable_lock
    def __eq__(self, other):
        if isinstance(other, Cache):
            return self.cache_items.__eq__(other.cache_items) and all(map(lambda a, b: a == b, self, other))
        return dict.__eq__(self.cache_items, other.cache_items)

    @_enable_thread_pool
    @_enable_lock
    def keys(self, timeout=2):
        return self.cache_items.keys()

    @_enable_thread_pool
    @_enable_lock
    def items(self, timeout=2):
        return self.cache_items.items()

    @_enable_thread_pool
    @_enable_lock
    def values(self, timeout=2):
        return self.cache_items.values()

    @_enable_lock
    def _generate_cache_instance_name(self, basic_name):
        name = basic_name + '-' + str(Cache.cache_instance_id_counter)
        Cache.cache_instance_id_counter += 1
        return name

    def _init_logger(self, level, name, filename, format):
        self.logger = logging.getLogger(self.__class__.__name__ + '<%s>' % name)
        if filename is not None:
            handler = logging.FileHandler(filename=filename)
        else:
            handler = logging.StreamHandler(sys.stdout)
        if format is not None:
            handler.setFormatter(logging.Formatter(format))
        handler.setLevel(level)
        self.logger.addHandler(handler)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
