# Import functions
import time
import os
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG, ERROR
from tybmilib import myfilename as mfn

#------------------------------------------------------------
# Read local file `config.ini`.
Local_mode = mfn.get_localmode()

def init_logger(exp_id, log_filename):
    """ログ設定の初期化

    Args:
        exp_id (str): 1st argument
        log_filename: 2nd argument
    Returns:
        logger: logger

    """
    log_name = os.path.splitext(os.path.basename(log_filename))[0]
    logger = getLogger(log_name).getChild(exp_id)
    logger.setLevel(DEBUG)
    
    # 出力先のログがない場合、ログを生成
    if not logger.hasHandlers():
        formatter = Formatter('%(asctime)s <%(levelname)s> : %(message)s')
        fileHandler = FileHandler(log_filename, mode='a')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    return logger

def logError(exp_id, msg, log_filename):
    """ログをERRORレベルで出力
    
    Args:
        exp_id (str): 1st argument
        msg (str): 2nd argument
        log_filename (str): 3rd argument
    Returns:
        None
    
    """
    log = init_logger(exp_id, log_filename)
    log.error(msg)


def logInfo(exp_id, msg, log_filename):
    """ログをINFOレベルで出力
    
    Args:
        exp_id (str): 1st argument
        msg (str): 2nd argument
        log_filename (str): 3rd argument
    Returns:
        None
    
    """
    log = init_logger(exp_id, log_filename)
    log.info(msg)


def logDebug(exp_id, msg, log_filename):
    """ログをDEBUGレベルで出力
    
    Args:
        exp_id (str): 1st argument
        msg (str): 2nd argument
        log_filename (str): 3rd argument
    Returns:
        None
    
    """
    log = init_logger(exp_id, log_filename)
    log.debug(msg)

class Lib_ParseError(Exception):
    """module内エラー出力用のクラス
    
    モジュール内で発生した固有の処理エラーに対し、指定のExceptionクラスを付与し、出力をするたためのクラス
    """
    pass


def raiseError(exp_id, error_msg, log_filename):
    """ログをエラーレベルで出力し、エラーを発生
    
    Args:
        exp_id (str): 1st argument
        error_msg (str): 2nd argument
        log_filename (str): 3rd argument
    Returns:
        None
    
    """
    logError(exp_id, error_msg, log_filename)
    raise Lib_ParseError(error_msg)