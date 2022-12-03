import logging
import os
from celery import Celery



formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)


def set_logger(logger):
    file_handler = logging.FileHandler(os.environ.get('LOG_FILE', 'logs.txt'))
    file_handler.setFormatter(formatter)
    """Setup logger."""
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def make_celery():
    celery = Celery(
        __name__,
        backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
        broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    )
    return celery