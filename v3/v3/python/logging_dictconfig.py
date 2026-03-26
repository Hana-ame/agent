import logging.config
import json

LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'simple': {'format': '%(asctime)s - %(levelname)s - %(message)s'},
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'level': 'INFO',
        },
    },
    'loggers': {
        'demo': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['console'],
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger('demo')
logger.debug('This is debug')
logger.info('This is info')
logger.warning('This is warning')
