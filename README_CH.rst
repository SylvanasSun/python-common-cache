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
.. image:: https://img.shields.io/github/commits-since/SylvanasSun/python-common-cache/0.1.svg
    :target: https://github.com/SylvanasSun/python-common-cache

\

.. _English: README.rst

该项目是一个基于内存的缓存组件，它是轻量级的、简单的和可自定义化的，你可以以一种非常简单的方法来实现你的需求。


Features
--------

- 开箱即用，没有复杂的配置代码，你可以通过默认配置来简单、轻松地使用缓存，但同时也支持自定义配置以满足需求，例如，自定义evict策略、清理策略以及是否使用线程池等等。

- 细粒度的过期时间控制，每个缓存实例对象都含有一个全局的过期时间，但你也可以通过函数put(key, value, expire)来设置局部的过期时间。

- 通过函数cache_loader(key)和cache_writer(key, value)以支持构建多级缓存系统，前者可以在缓存未命中的情况下从其他缓存系统或者数据源读取缓存，后者则可以将查询到的结果写入到另一个缓存系统或者数据源，以上两个函数都需要你自己实现并配置到缓存实例对象中。

- 当缓存的容量已满时，默认使用LRU(Least-Recently-Used)策略进行回收旧数据，并且还提供了另外两种其他的策略：FIFO(First-In-First-Out)与LFU(Least-Frequently-Used)。

- 通过函数replace_evict_func(func)与replace_cleanup_func()支持在运行时进行动态替换驱逐策略与清理策略。

- 在缓存实例对象实例化时创建一个用于定期清理无效缓存项的守护进程，而且缓存实例对象在每次使用get()/put()/pop()操作之后,都会调用清理函数进行清理。

- 每个缓存实例对象默认都有一个用于提高吞吐量的线程池，你可以在创建缓存实例时选择不使用线程池，例如“cache = Cache(enable_thread_pool = False)”，也可以在运行时对线程池进行动态开关。

- 记录了每个缓存项的统计信息，包括命中次数、命中率、未命中次数、expire时间、剩余时间、创建时间和key与value，你可以通过调用statistic_record()函数来获得这些信息。

Usage
-----

首先你需要通过pip进行安装。

::

    pip install python-common-cache

有两种使用方法, 第一种就是直接使用缓存实例，就像使用一个dict一样:

::

    cache = Cache(expire=10)
    cache['key'] = 'data'
    def foo():
        # cache hit and return data from the cache
        if cache['key'] is not None:
            return cache['key']
        # cache miss and return data from a data source or service
        ....

第二种是通过缓存实例提供的装饰器，这种方法更加方便且实用:

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

更多的用法请阅读Cache类的源码，其中有非常多的使用案例代码以供参考。