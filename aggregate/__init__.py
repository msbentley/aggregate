#!/usr/bin/env python
# encoding: utf-8
"""
__init__.py

"""

__all__ = ['builder', 'simulation']

from aggregate import *

# Set up the root logger

import logging, sys
logging.basicConfig(format='%(levelname)s (%(name)s): %(message)s',
                     level=logging.INFO, stream=sys.stdout)

