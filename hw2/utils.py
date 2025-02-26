import time
import logging

from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Define color mapping for different log levels
COLORS = {
    logging.INFO: Fore.GREEN,
    logging.DEBUG: Fore.CYAN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
}

# Custom formatter for colored logs
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelno, "")
        log_message = super().format(record)
        return f"{color}{log_message}{Style.RESET_ALL}"

class Logger: 
    def __init__(self, level: int = logging.INFO) -> None:
        self.logger = logging.getLogger("green_logger")
        self.logger.setLevel(level=level)
        ch = logging.StreamHandler()
        ch.setFormatter(ColoredFormatter("%(levelname)s: %(message)s"))
        self.logger.addHandler(ch)

    def set_level(self, level: int) -> None:
        self.logger.setLevel(level)
    
    def info(self, message: str) -> None:
        self.logger.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - {message}")

    def debug(self, message: str) -> None:
        self.logger.debug(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - {message}")

    def warning(self, message: str) -> None:
        self.logger.warning(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - {message}")

    def error(self, message: str) -> None:
        self.logger.error(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] - {message}")
