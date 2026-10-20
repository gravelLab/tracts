from __future__ import annotations
import logging
import logging.handlers
import inspect
from pathlib import Path
import sys
import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for type checking: a top-level import would create a circular import,
    # since tracts.driver_utils (via tracts.population) imports from this module.
    from tracts.driver_utils import InferenceConfig

# ---------- Logger setup ----------

LOGGER_NAME = "tracts"

def _get_formatter():
    return logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def open_logger():
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

def initialize_tracts(driver_spec: InferenceConfig, driver_filename: str) -> tuple[logging.Logger, Path, Path]:
    """
    Sets up the tracts logger for a run and resolves the log file path from the driver specification. Also
    prints initial information about the run to both the console and the log file.

    Initializes the logger and its in-memory buffer via :func:`~tracts.logs.open_logger`, determines the
    output directory (creating it if it does not already exist; the ``{date}`` placeholder in
    ``driver_spec.output.output_directory`` is filled in with the current timestamp), resolves the full
    path of the log file (placing it inside the output directory if ``driver_spec.output.log_filename`` is
    a bare filename with no parent directory), and attaches a file handler to the logger via
    :func:`~tracts.logs.set_log_file`.

    Parameters
    ----------
    driver_spec: InferenceConfig
        The configuration for the inference process, as specified in the driver file. If
        ``driver_spec.output.log_filename`` is not set, defaults to ``"tracts.log"``. If
        ``driver_spec.output.output_directory`` is not set, defaults to the current working directory;
        in that case, ``driver_spec.output.output_directory`` is updated in place to record the resolved
        directory.
    driver_filename: str
        The path to the driver file, used for logging and printing initial information about the run.

    Returns
    -------
    tuple[logging.Logger, Path, Path]
        A tuple ``(logger, log_full_path, output_dir)``, where ``logger`` is the configured tracts logger
        (with the file handler already attached), ``log_full_path`` is the resolved path of the log file,
        and ``output_dir`` is the resolved output directory (created if it did not already exist).
    """
    logger, memory_handler = open_logger()
    if driver_spec.output.log_filename:
        log_filename = Path(driver_spec.output.log_filename)
    else:
        log_filename = Path("tracts.log")
        logger.warning(f"No log filename specified in driver file. Defaulting to {log_filename} in the working directory.")
    
    if not driver_spec.output.output_directory:
        logger.warning("No output directory specified in driver file. Defaulting to current working directory.")
        output_dir = Path.cwd()
        driver_spec.output.output_directory = str(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_output_directory =  driver_spec.output.output_directory.format(date=timestamp)
        output_dir = Path(formatted_output_directory)
 
    if not os.path.exists(output_dir): # Create output directory if it doesn't exist 
        os.makedirs(output_dir)
    
    log_full_path = Path(log_filename)
    if not log_full_path.is_absolute() and log_full_path.parent == Path("."): # If log_filename is a relative path without directories, save it in the output directory. Otherwise, save it in the specified path (which may be absolute or relative with directories).
        log_full_path = Path(output_dir) / log_full_path

    set_log_file(log_filename=log_full_path,
                memory_handler=memory_handler)

    logger.info(f"Running tracts 2.0 with driver file: {driver_filename}")
    output_message = f"Results will be written to: {output_dir}."
    logger_message = f"Using log file: {log_full_path}."
    tracts_below_cm_message = f'excluding_tracts_below set to {driver_spec.optim.exclude_tracts_below_cm} cM.'
    re_optimization_message = f"Re-optimization will be performed until convergence or maximum {driver_spec.optim.n_reoptimizations} times." if driver_spec.optim.n_reoptimizations > 0 else "Re-optimization will not be performed."

    # ------ Print initial information -------
    print('------------------------------------------------------------------------------------------------\n')
    print('Running tracts 2.0 with driver file:', driver_filename,'\n')
    print('------------------------------------------------------------------------------------------------\n')   
    for message in (output_message, logger_message, tracts_below_cm_message, re_optimization_message):
        print(message)
        logger.info(message)

    return logger, log_full_path, output_dir