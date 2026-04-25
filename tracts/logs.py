import logging
import inspect      
from pathlib import Path
import sys

# ---------- Logger setup ----------

LOGGER_NAME = "tracts"

def _get_formatter():
    return logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def setup_logger():
    logger = logging.getLogger(LOGGER_NAME)

    if logger.handlers:
        memory_handler = next(
            (h for h in logger.handlers if isinstance(h, logging.handlers.MemoryHandler)),
            None
        )
        return logger, memory_handler

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = _get_formatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    memory_handler = logging.handlers.MemoryHandler(
        capacity=10000,
        flushLevel=logging.CRITICAL,
        target=None
    )
    memory_handler.setLevel(logging.INFO)

    logger.addHandler(stream_handler)
    logger.addHandler(memory_handler)

    return logger, memory_handler


def set_log_file(log_filename: str | Path, memory_handler):
    """
    Sets up logging to a file. If the logger already has handlers,
    it will buffer log records in memory until the file handler is added, 
    at which point it will flush the buffered records to the file.
    
    Parameters
    ----------
    log_filename: str | Path
        The name of the log file to write to.
    memory_handler: logging.handlers.MemoryHandler
        The memory handler used to buffer log records until the file handler is added.      
    """

    logger = logging.getLogger(LOGGER_NAME)
    formatter = _get_formatter()

    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    if memory_handler is not None:
        memory_handler.setTarget(file_handler)
        memory_handler.flush()
        logger.removeHandler(memory_handler)
        memory_handler.close()

    return file_handler

def get_current_func_info():
    frame = inspect.currentframe().f_back  # One level up: the caller
    file_name = frame.f_code.co_filename
    func_name = frame.f_code.co_name
    line_number = frame.f_lineno
    return file_name, func_name, line_number