#-*- coding: utf-8 -*-
"""
@author: TOYOBO CO., LTD.
"""
# Import functions
import time
import os
import sys
import pathlib
from pathlib import Path
from datetime import datetime
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG, ERROR
from functools import wraps

#------------------------------------------------------------
# 出力レベルはconfigで外出しである必要
def create_logger(exp_version):
    new_path = 'logs' #フォルダ名
    if not os.path.exists(new_path):#ディレクトリがなかったら
        os.mkdir(new_path)#作成したいフォルダ名を作成
    path = os.getcwd()
    
    log_file = pathlib.Path(path+'/logs/' + '{}.log'.format(exp_version + '_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    
    # logger
    logger_ = getLogger(__name__)
    logger_.setLevel(DEBUG)

    # formatter
    fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    # file handler
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    fh.setFormatter(fmr)

    # stream handler
    ch = StreamHandler()
    ch.setLevel(ERROR)
    ch.setFormatter(fmr)

    logger_.addHandler(fh)
    logger_.addHandler(ch)

def logger():
    return getLogger(__name__)

def stop_watch(VERSION):
    def _stop_watch(func):
        @wraps(func)
        def wrapper(*args, **kargs):
            start = time.time()

            result = func(*args, **kargs)

            elapsed_time = int(time.time() - start)
            minits, sec = divmod(elapsed_time, 60)
            hour, minits = divmod(minits, 60)

            get_logger(VERSION).info("[elapsed_time]\t>> {:0>2}:{:0>2}:{:0>2}".format(hour, minits, sec))
        return wrapper
    return _stop_watch