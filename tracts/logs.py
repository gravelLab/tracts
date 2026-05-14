import logging
import logging.handlers
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

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = _get_formatter()

    # Add stream handler only once
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )

    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Remove any stale MemoryHandler from a previous incomplete run
    for handler in list(logger.handlers):
        if isinstance(handler, logging.handlers.MemoryHandler):
            logger.removeHandler(handler)
            handler.close()

    # Always create a fresh MemoryHandler for the current run
    memory_handler = logging.handlers.MemoryHandler(
        capacity=10000,
        flushLevel=logging.CRITICAL,
        target=None,
    )
    memory_handler.setLevel(logging.INFO)
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
    log_filename = Path(log_filename)

    # Remove previous file handlers
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

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

def close_log_file(log_filename: str | Path):
    """
    Closes the log file handler associated with the given log filename. This is important to ensure
    that all log records are flushed to the file and that file handles are not left open,
    which can lead to issues on some operating systems.

    Parameters
    ----------
    log_filename: str | Path
        The name of the log file whose handler should be closed.
    """

    logger = logging.getLogger(LOGGER_NAME)
    log_path = Path(log_filename).resolve()

    file_handlers_to_close = [
        h for h in logger.handlers
        if isinstance(h, logging.FileHandler)
        and Path(h.baseFilename).resolve() == log_path
    ]

    for file_handler in file_handlers_to_close:
        # Flush memory handlers that target this file handler
        for handler in logger.handlers[:]:
            if (
                isinstance(handler, logging.handlers.MemoryHandler)
                and handler.target is file_handler
            ):
                handler.flush()
                logger.removeHandler(handler)
                handler.close()

        file_handler.flush()
        logger.removeHandler(file_handler)
        file_handler.close()


def get_current_func_info():
    frame = inspect.currentframe().f_back  # One level up: the caller
    file_name = frame.f_code.co_filename
    func_name = frame.f_code.co_name
    line_number = frame.f_lineno
    return file_name, func_name, line_number