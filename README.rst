python-common-cache
-------------------

.. image:: https://img.shields.io/github/forks/SylvanasSun/python-common-cache.svg?style=social&label=Fork
    :target: https://github.com/SylvanasSun/python-common-cache
.. image:: https://img.shields.io/github/stars/SylvanasSun/python-common-cache.svg?style=social&label=Stars
    :target: https://github.com/SylvanasSun/python-common-cache
.. image:: https://img.shields.io/github/watchers/SylvanasSun/python-common-cache.svg?style=social&label=Watch
    :target: https://github.com/SylvanasSun/python-common-cache
.. image:: https://img.shields.io/github/followers/SylvanasSun.svg?style=social&label=Follow
    :target: https://github.com/SylvanasSun/python-common-cache

\

.. image:: https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php
    :target: LICENSE
.. image:: https://travis-ci.org/SylvanasSun/python-common-cache.svg?branch=master
    :target: https://travis-ci.org/SylvanasSun/python-common-cache
.. image:: https://img.shields.io/pypi/pyversions/python-common-cache.svg
    :target: https://pypi.python.org/pypi/python-common-cache
.. image:: https://img.shields.io/pypi/v/python-common-cache.svg
    :target: https://pypi.python.org/pypi/python-common-cache
.. image:: https://img.shields.io/badge/version-0.1-brightgreen.svg
    :target: https://pypi.python.org/pypi/python-common-cache
.. image:: https://img.shields.io/github/release/SylvanasSun/python-common-cache.svg
    :target: https://github.com/SylvanasSun/python-common-cache
.. image:: https://img.shields.io/github/tag/SylvanasSun/python-common-cache.svg
    :target: https://github.com/SylvanasSun/python-common-cache
.. image:: https://img.shields.io/github/issues/SylvanasSun/python-common-cache.svg
    :target: https://github.com/SylvanasSun/python-common-cache

\

简体中文_

.. _简体中文: README_CH.rst


This project is an cache component based on the memory and it is lightweight, simple and customizable, you can implement a cache that your needs in a very simple way.


Features
--------

- Out-of-the-box, there is no complex configuration code you can easily use it by default configuration but also support a customized configuration for your needs such as customized evict strategy, cleanup strategy and whether enable thread pool and so on.

- Fine-grained control for expiration time, each cache instance has a global expiration time but you can also set the local expiration time of the key by put(key, value, expire) function.

- Supports to build a multi-level cache system by cache_loader(key) and cache_writer(key, value), the former will can load cache from other cache system or data source when the cache is miss and the latter will can take the results write to another cache system or data source, above two function needs you to implement and configure to the cache instance.

- Default use LRU(Least-Recently-Used) strategy for recycle old cache item when the capacity of the cache is full and also provide two other kinds of the evict strategy FIFO(First-In-First-Out) and LFU(Least-Frequently-Used).

- Supports dynamic replacement evict function and cleanup function in the runtime by replace_evict_func(func) and replace_cleanup_func(func).

- Create a daemon when cache instance initialize and it will to regularly clean up invalid cache item by invoking cleanup() and cache instance will be invoked cleanup() when each use get()/put()/pop() after.

- Each cache instance default has a thread pool for improving throughput, you can choose not use thread pool when creating cache instance such as "cache = Cache(enable_thread_pool=False)" and can also dynamic control enable or unable thread pool when runtime.

- Have recorded statistics information of each cache item and the information is included hit counts, hit rate, miss counts, expiration time, remaining survival time, birthday and key and value, you can get those by invoking function statistic_record().

Usage
-----

First you need to install it.

::

    pip install python-common-cache

Have two way for use cache, the first is direct use it is like use a dictionary:

::

    cache = Cache(expire=10)
    cache['key'] = 'data'
    def foo():
        # cache hit and return data from the cache
        if cache['key'] is not None:
            return cache['key']
        # cache miss and return data from a data source or service
        ....

The second way is to use decorator and this way more convenient:

::

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

For more usage please read the source code of the class Cache and have many document test example code for reference.