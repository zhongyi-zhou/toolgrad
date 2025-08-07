import logging
import pytz
import asyncio
import functools
import collections
import time
import datetime
import os

_LOG_TIME_DEQUE = collections.defaultdict(lambda: collections.deque(maxlen=100))

_JST = datetime.timezone(datetime.timedelta(hours=+9), 'JST')
_TOKYO_TIME = pytz.timezone('Asia/Tokyo')

_LOG_PATH = "./logs/" + datetime.datetime.now(_TOKYO_TIME).strftime(
    '%Y-%m-%d/%H:%M:%S') + '.log'


def log_time(func):
  if asyncio.iscoroutinefunction(func):

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
      start_time = time.perf_counter()
      result = await func(*args, **kwargs)
      elapsed_time = time.perf_counter() - start_time

      # Append the elapsed time to the deque for this function.
      _LOG_TIME_DEQUE[func.__name__].append(elapsed_time)
      # Compute rolling average using the values in the deque.
      avg_time = sum(_LOG_TIME_DEQUE[func.__name__]) / len(
          _LOG_TIME_DEQUE[func.__name__])

      logging.info(
          f"{func.__name__} took {elapsed_time:.4f} sec | Avg: {avg_time:.4f} sec "
          f"(Last {len(_LOG_TIME_DEQUE[func.__name__])} runs)")
      return result

    return async_wrapper
  else:

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
      start_time = time.perf_counter()
      result = func(*args, **kwargs)
      elapsed_time = time.perf_counter() - start_time

      # Append the elapsed time to the deque for this function.
      _LOG_TIME_DEQUE[func.__name__].append(elapsed_time)
      # Compute rolling average using the values in the deque.
      avg_time = sum(_LOG_TIME_DEQUE[func.__name__]) / len(
          _LOG_TIME_DEQUE[func.__name__])

      logging.info(
          f"{func.__name__} took {elapsed_time:.4f} sec | Avg: {avg_time:.4f} sec "
          f"(Last {len(_LOG_TIME_DEQUE[func.__name__])} runs)")
      return result

    return sync_wrapper


class TokyoFormatter(logging.Formatter):
  """Custom formatter to use Tokyo timezone for logging timestamps"""

  def converter(self, timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp,
                                         tz=pytz.timezone("Asia/Tokyo"))
    return dt

  def formatTime(self, record, datefmt=None):
    dt = self.converter(record.created)
    return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


def config_logging(
    level=logging.INFO,
    log_file_path: str = 'my_log_file.log',
    retention_days: int = 14,
    logger_name: str = None,  # Optional logger name for thread-specific loggers
):
  """Configures a logger that writes to a TimedRotatingFileHandler.
    
    Args:
        level (int): The logging level.
        log_file_path (str): The file path for the log file.
        retention_days (int): How many days to keep rotated log files.
        logger_name (str, optional): If provided, a logger with this name is used.
                                     Otherwise, the root logger is configured.
                                     
    Returns:
        logging.Logger: The configured logger.
    """
  # Get a logger. If logger_name is provided, this returns a new/reused logger
  logger = logging.getLogger(logger_name)

  # Ensure the directory for the log file exists.
  os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

  # Create a handler that rotates logs daily.
  log_handler = logging.handlers.TimedRotatingFileHandler(
      log_file_path, when="D", interval=1, backupCount=retention_days)

  # Set up the formatter (using your custom TokyoFormatter)
  formatter = TokyoFormatter('%(asctime)s - %(levelname)s - %(message)s')
  log_handler.setFormatter(formatter)

  # Clear any existing handlers from this logger to prevent duplicate logs.
  if logger.hasHandlers():
    logger.handlers.clear()

  # Add our new handler.
  logger.addHandler(log_handler)
  logger.setLevel(level)

  return logger
