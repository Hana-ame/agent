import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'level': record.levelname,
            'message': record.getMessage(),
            'timestamp': self.formatTime(record)
        }
        return json.dumps(log_data)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logger.info("User login", extra={'user': 'alice'})
logger.warning("Disk space low", extra={'used': '95%'})
