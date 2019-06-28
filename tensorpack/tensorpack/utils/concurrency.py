# -*- coding: UTF-8 -*-
# File: concurrency.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
# Credit belongs to Xinyu Zhou

import threading
import multiprocessing
import atexit
import bisect
from contextlib import contextmanager
import signal
import weakref
import six
if six.PY2:
    import subprocess32 as subprocess
else:
    import subprocess
from six.moves import queue

from . import logger

__all__ = ['StoppableThread', 'LoopThread', 'ensure_proc_terminate',
           'OrderedResultGatherProc', 'OrderedContainer', 'DIE',
           'mask_sigint', 'start_proc_mask_signal']

class StoppableThread(threading.Thread):
    """
    A thread that has a 'stop' event.
    """
    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop_evt = threading.Event()

    def stop(self):
        """ stop the thread"""
        self._stop_evt.set()

    def stopped(self):
        """ check whether the thread is stopped or not"""
        return self._stop_evt.isSet()

    def queue_put_stoppable(self, q, obj):
        """ put obj to queue, but will give up if the thread is stopped"""
        while not self.stopped():
            try:
                q.put(obj, timeout=5)
                break
            except queue.Full:
                pass

    def queue_get_stoppable(self, q):
        """ take obj from queue, but will give up if the thread is stopped"""
        while not self.stopped():
            try:
                return q.get(timeout=5)
            except queue.Empty:
                pass

class LoopThread(StoppableThread):
    """ A pausable thread that simply runs a loop"""
    def __init__(self, func, pausable=True):
        """
        :param func: the function to run
        """
        super(LoopThread, self).__init__()
        self._func = func
        self._pausable = pausable
        if pausable:
            self._lock = threading.Lock()
        self.daemon = True

    def run(self):
        while not self.stopped():
            if self._pausable:
                self._lock.acquire()
                self._lock.release()
            self._func()

    def pause(self):
        assert self._pausable
        self._lock.acquire()

    def resume(self):
        assert self._pausable
        self._lock.release()


class DIE(object):
    """ A placeholder class indicating end of queue """
    pass

def ensure_proc_terminate(proc):
    if isinstance(proc, list):
        for p in proc:
            ensure_proc_terminate(p)
        return

    def stop_proc_by_weak_ref(ref):
        proc = ref()
        if proc is None:
            return
        if not proc.is_alive():
            return
        proc.terminate()
        proc.join()

    assert isinstance(proc, multiprocessing.Process)
    atexit.register(stop_proc_by_weak_ref, weakref.ref(proc))


@contextmanager
def mask_sigint():
    sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    yield
    signal.signal(signal.SIGINT, sigint_handler)

def start_proc_mask_signal(proc):
    if not isinstance(proc, list):
        proc = [proc]

    with mask_sigint():
        for p in proc:
            p.start()

def subproc_call(cmd, timeout=None):
    try:
        output = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT,
                shell=True, timeout=timeout)
        return output
    except subprocess.TimeoutExpired as e:
        logger.warn("Command timeout!")
        logger.warn(e.output)
    except subprocess.CalledProcessError as e:
        logger.warn("Commnad failed: {}".format(e.returncode))
        logger.warn(e.output)

class OrderedContainer(object):
    """
    Like a priority queue, but will always wait for item with index (x+1) before producing (x+2).
    """
    def __init__(self, start=0):
        self.ranks = []
        self.data = []
        self.wait_for = start

    def put(self, rank, val):
        idx = bisect.bisect(self.ranks, rank)
        self.ranks.insert(idx, rank)
        self.data.insert(idx, val)

    def has_next(self):
        if len(self.ranks) == 0:
            return False
        return self.ranks[0] == self.wait_for

    def get(self):
        assert self.has_next()
        ret = self.data[0]
        rank = self.ranks[0]
        del self.ranks[0]
        del self.data[0]
        self.wait_for += 1
        return rank, ret

class OrderedResultGatherProc(multiprocessing.Process):
    """
    Gather indexed data from a data queue, and produce results with the
    original index-based order.
    """
    def __init__(self, data_queue, nr_producer, start=0):
        """
        :param data_queue: a multiprocessing.Queue to produce input dp
        :param nr_producer: number of producer processes. Will terminate after receiving this many of DIE sentinel.
        :param start: the first task index
        """
        super(OrderedResultGatherProc, self).__init__()
        self.data_queue = data_queue
        self.ordered_container = OrderedContainer(start=start)
        self.result_queue = multiprocessing.Queue()
        self.nr_producer = nr_producer

    def run(self):
        nr_end = 0
        try:
            while True:
                task_id, data = self.data_queue.get()
                if task_id == DIE:
                    self.result_queue.put((task_id, data))
                    nr_end += 1
                    if nr_end == self.nr_producer:
                        return
                else:
                    self.ordered_container.put(task_id, data)
                    while self.ordered_container.has_next():
                        self.result_queue.put(self.ordered_container.get())
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    def get(self):
        return self.result_queue.get()
