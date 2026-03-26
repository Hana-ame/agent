try:
    import structlog
except ImportError:
    print("structlog not installed, skipping")
    exit(0)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()
logger.info("User logged in", user_id=123)
logger.warning("Disk usage high", usage=85)
