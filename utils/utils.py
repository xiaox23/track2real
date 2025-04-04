import copy

from datetime import datetime
import os, sys
import threading
import time

import numpy as np


class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def try_make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


class ThreadSafeContainer():
    def __init__(self, max_size=100):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.lock = threading.Lock()
        self.current_size = 0

    def get(self, idx_list, numpy=True):
        self.lock.acquire()
        if type(idx_list) is list:
            ret = []
            for idx in idx_list:
                local_idx = (idx + self.ptr) % len(self.storage)
                ret.append(copy.deepcopy(self.storage[local_idx]))
            if numpy:
                ret = np.array(ret)
        else:
            local_idx = (idx_list + self.ptr) % len(self.storage)
            ret = copy.deepcopy(self.storage[local_idx])
        self.lock.release()
        return ret

    def put(self, data):
        self.lock.acquire()
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)
        self.current_size = len(self.storage)
        self.lock.release()

    def clear(self):
        self.lock.acquire()
        self.storage.clear()
        self.current_size = len(self.storage)
        self.lock.release()

    def empty(self):
        return self.current_size == 0

    def check_shape(self):
        all_same_shape = len(set(array.shape for array in self.storage)) == 1
        return all_same_shape

def get_ms():
    """get millisecond of now in string of length 3"""
    a = str(int(time.time() * 1000) % 1000)
    if len(a) == 1:
        return '00' + a
    if len(a) == 2:
        return '0' + a
    return a


def get_time():
    """get time in format HH:MM:SS:MS"""
    now = "{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
    return now + '.' + get_ms()


def dump_args(f, arg):
    dict = arg.__dict__
    time = get_time()
    f.write('--------------------Arguments--------------------\n')
    f.write('Begin at time : ' + time + '\n')
    for key, value in dict.items():
        f.write("{:>40}: {:<100}\n".format(key, value))
    f.write('-----------------------End-----------------------\n')


def dump_args_to_tensorboard(tensorboard_writer, arg, global_step=0):
    dict = arg.__dict__
    time = get_time()
    string = f'Begin at time : {time} \n'
    for key, value in dict.items():
        try:
            str_append = "{:>50}: {:<100}\n".format(key, value)
            string += str_append
        except:
            pass
    tensorboard_writer.add_text(tag="ARGUMENTS", text_string=string, global_step=global_step)


def dump_dict_to_tensorboard(tensorboard_writer, target_dict, global_step=0):
    time = get_time()
    string = f"Log time : {time} \n"

    def append_dict_to_string(target_string, target_dict, prefix=''):
        for key, value in target_dict.items():
            if type(value) is dict:
                target_string = append_dict_to_string(target_string, value, prefix=f"{key}.")
            else:
                try:
                    str_append = "{:>50}: {:<100}\n".format(prefix + key, value)
                    target_string += str_append
                except:
                    pass
        return target_string

    string = append_dict_to_string(string, target_dict)
    tensorboard_writer.add_text(tag="ARGUMENTS", text_string=string, global_step=global_step)


def copy_args(source, target):
    """
    arg is the parsed arguments
    param is SimParams object
    """
    source_dict = source.__dict__
    target_dict = target.__dict__
    for key, value in source_dict.items():
        if key in target_dict.keys():
            target.__setattr__(key, value)

