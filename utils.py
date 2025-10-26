import logging
from typing import Optional

import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

# Configure logging to debug level
logging.basicConfig(level=logging.DEBUG)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def make_retry_session(
    total: int = 5,
    backoff_factor: float = 0.5,
    status_forcelist: Optional[list] = None,
    connect_retries: int = 2,
) -> requests.Session:
    if status_forcelist is None:
        status_forcelist = [429, 500, 502, 503, 504]

    retry = Retry(
        total=total,
        read=total,
        connect=connect_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


