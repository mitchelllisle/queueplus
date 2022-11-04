__version__ = '0.4.0'
__author__ = 'Mitchell Lisle'
__email__ = 'm.lisle90@gmail.com'

from queueplus.aioqueue import AioQueue, DataT, TypedAioQueue
from queueplus.violations import DiscardOnViolation, RaiseOnViolation, ViolationStrategy
