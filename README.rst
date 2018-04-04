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


python-common-cache is a cache component based in memory. It is lightweight, simple, and customizable. Implementing a cache that fits your needs is as simple as installing and importing.


Features
--------

- Complex configuration code is not necessary. The default configuration is suited for most use cases but there is also support for customized configuration in regards to: eviction strategy, cleanup strategy, and whether to enable thread pool and so on.

- Fine-grained control of expiration time. Each cache instance has a global expiration time but can also set the local expiration time of the key by using ``put(key, value, expire)``.

- Supports building a multi-level cache system using the ``cache_loader(key)`` and the ``cache_writer(key, value)``. The ``cache_loader(key)`` can load a cache from another cache system or data source when there is a cache miss. The ``cache_writer(key, value)`` can take the results and write to another cache system or data source. The ``cache_loader(key)`` and ``cache_writer(key, value)``  needs you to implement and configure to the cache instance.

- LRU (Least-Recently-Used) strategy for recycling old cache items when the capacity of the cache is full. FIFO (First-In-First-Out) and LFU (Least-Frequently-Used) can also be used instead.

- Supports dynamic replacement eviction by using ``replace_evict_func(func)`` and a cleanup function using ``replace_cleanup_func(func)``.

- Creates a daemon when a cache instance is initialized and will regularly clean up invalid cache items by invoking ``cleanup()``. The cache instance will invoke ``cleanup()`` after using ``get()/put()/pop()``.

- Each cache instance has a thread pool for improving throughput, which can be disabled by passing ``enable_thread_pool=False`` to the ``Cache()`` function which can also invoke dynamic control or by disabling the thread pool at runtime

- ``statistic_record()`` outputs the recorded statistics of each cache item including: hit counts, hit rate, miss counts, expiration time, remaining survival time, birthday, and key and value

Installation
-----

Install using pip

::

    pip install python-common-cache
    
Usage
-----

There are two ways to use the cache

The first is to use it like a dictionary:

::

    cache = Cache(expire=10)
    cache['key'] = 'data'
    def foo():
        # cache hit and return data from the cache
        if cache['key'] is not None:
            return cache['key']
        # cache miss and return data from a data source or service
        ....

The second way is to use a decorator which is more convenient:

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

For more usage examples please read the source code of the Cache class where there are many examples for reference.
