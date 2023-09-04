import functools


def count_time(func):
    """装饰器：计算程序运行时间"""
    from datetime import datetime

    @functools.wraps(func)  # 保留被装饰函数信息
    def wrapper(*args, **kw):
        start_time = datetime.now()
        res = func(*args, **kw)
        print("[%s] RUN TIME: %s" % (func.__name__, str(datetime.now() - start_time)))
        return res  # 返回被装饰函数的返回值

    return wrapper
