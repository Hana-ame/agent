import logging
import logging.config
import json
from pathlib import Path

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(levelname)s %(name)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'app.log',
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 3,
            'formatter': 'standard',
        }
    },
    'loggers': {
        'app': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['console']
    }
}

# 检查 python-json-logger 是否安装
try:
    import pythonjsonlogger
    LOGGING_CONFIG['formatters']['json']['()'] = 'pythonjsonlogger.jsonlogger.JsonFormatter'
    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False
    LOGGING_CONFIG['formatters'].pop('json', None)
    print("python-json-logger not installed, JSON logging disabled")

logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger('app')
logger.debug("This is a debug message")
logger.info("Application started")
logger.warning("Disk space low")
logger.error("Failed to connect to data
