from tybmilib import logtest

def temp_print(exp_id):
    log_filename = "temp.log"
    logtest.init_log(exp_id, log_filename)
    logtest.aplInfo("temp {}".format(exp_id), log_filename)
    
    return "temp"

def temp_id_print(exp_id):
    log_filename = "temp.log"
    logtest.aplInfo("temp id {}".format(exp_id), log_filename)
    
    return "temp {}".format(exp_id)