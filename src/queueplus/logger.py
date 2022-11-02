import logging
import os

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger(__name__)
