import logging
from typing import Optional

def get_logger(name: str, level: int = logging.INFO, fmt: Optional[str] = None) -> logging.Logger:
    """Create and configure a logger instance.
    
    Returns a logger with a StreamHandler configured. If the logger already
    has handlers, returns the existing logger without reconfiguration.
    
    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).
        fmt: Custom format string for log messages. If None, uses default format.
        
    Returns:
        Configured Logger instance ready for use.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt or "%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
