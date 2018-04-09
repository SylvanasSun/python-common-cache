#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import logging
import sys
import threading
from queue import Queue


def get_function_signature(func):
    """
    Return the signature string of the specified function.

    >>> def foo(name): pass
    >>> get_function_signature(foo)
    'foo(name)'
    >>> something = 'Hello'
    >>> get_function_signature(something)
    Traceback (most recent call last):
        ...
    TypeError: The argument must be a function object: None type is <class 'str'>
    """
    if func is None:
        return 'Function is None'

    try:
        func_name = func.__name__
    except AttributeError:
        func_name = 'None'

    if not inspect.isfunction(func):
        raise TypeError('The argument must be a function object: %s type is %s' % (func_name, type(func)))

    return func_name + str(inspect.signature(func))


def init_logger(self, level, name, filename, format):
    logger = logging.getLogger(self.__class__.__name__ + '<%s>' % name)
    if filename is not None:
        handler = logging.FileHandler(filename=filename)
        if format is not None:
            handler.setFormatter(logging.Formatter(format))
        handler.setLevel(level)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class RWLock(object):
    """
    The class RWLock represent a simple reader-writer lock, some readers can hold the lock simultaneously
    and writer locks have priority over reads to prevent write starvation.

    Reading and writing will cause blockages but reading and reading do not cause blockages,
    use a thread-safe queue for fairly schedule writers and avoid the starvation
    that can occur when a lot of writers waiting.
    """

    def __init__(self, max_reader_concurrency=sys.maxsize):
        self.mutex = threading.RLock()
        self.writers_waiting = 0
        self.writers_waiting_queue = Queue()
        self.rwlock = 0
        self.readers_ok = threading.Condition(self.mutex)
        self.max_reader_concurrency = max_reader_concurrency

    def acquire(self, only_read=True):
        if only_read:
            return self.acquire_reader
        else:
            return self.acquire_writer

    def acquire_reader(self):
        """
        Acquire a read lock, several threads can hold this type of lock.
        """
        with self.mutex:
            while self.rwlock < 0 or self.rwlock == self.max_reader_concurrency or self.writers_waiting:
                self.readers_ok.wait()
            self.rwlock += 1

    def acquire_writer(self):
        """
        Acquire a write lock, only one thread can hold this lock
        and only when no read locks are also held.
        """
        with self.mutex:
            while self.rwlock != 0:
                self._writer_wait()
            self.rwlock = -1

    def promote(self):
        """
        Promote an already acquired read lock to a write lock
        notice: this function can easily cause a deadlock!!!!!!!!!
        """
        with self.mutex:
            self.rwlock -= 1
            while self.rwlock != 0:
                self._writer_wait()
            self.rwlock = -1

    def demote(self):
        """
        Demote an already acquired write lock to a read lock
        """
        with self.mutex:
            self.rwlock = 1
            self.readers_ok.notify_all()

    def release(self):
        with self.mutex:
            if self.rwlock < 0:
                self.rwlock = 0
            else:
                self.rwlock -= 1
            wake_writers = self.writers_waiting and self.rwlock == 0
            wake_readers = self.writers_waiting == 0

        if wake_writers:
            writers_ok = self.writers_waiting_queue.get_nowait()
            with writers_ok:
                writers_ok.notify()
        elif wake_readers:
            with self.readers_ok:
                self.readers_ok.notify_all()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    __enter__ = acquire_reader

    def _writer_wait(self):
        self.writers_waiting += 1
        writers_ok = threading.Condition(self.mutex)
        self.writers_waiting_queue.put(writers_ok)
        writers_ok.wait()
        self.writers_waiting -= 1


if __name__ == '__main__':
    import doctest

    doctest.testmod()
