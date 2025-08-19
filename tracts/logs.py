import logging


def show_INFO(module=None):
    try:
        module.logger.setLevel(logging.INFO)
        logging.info('')
        module.logger.info(f' INFO logs enabled for this module.')
    except Exception as ex:
        print(f"Logging exception {ex}")
        pass
