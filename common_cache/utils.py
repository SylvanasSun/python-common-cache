#!/usr/bin/env python
# -*- coding: utf-8 -*-
import inspect
import logging


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


if __name__ == '__main__':
    import doctest

    doctest.testmod()
