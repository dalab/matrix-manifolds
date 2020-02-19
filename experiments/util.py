import time


class Timer(object):

    def __init__(self, msg=None):
        self.msg = msg

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.msg is not None:
            print('time({}): {:.2f}s'.format(self.msg, self.interval))
