import logging
import inspect

def show_INFO(module=None):
    try:
        module.logger.setLevel(logging.INFO)
        logging.info('')
        module.logger.info(f' INFO logs enabled for this module.')
    except Exception as ex:
        print(f"Logging exception {ex}")
        pass

def get_current_func_info():
    frame = inspect.currentframe().f_back  # One level up: the caller
    file_name = frame.f_code.co_filename
    func_name = frame.f_code.co_name
    line_number = frame.f_lineno
    return file_name, func_name, line_number