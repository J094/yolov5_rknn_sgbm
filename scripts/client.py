#!/usr/bin/env python
# J094
"""
Client test
"""

from utils.rosutils import rosClient


if __name__ == '__main__':
    c = rosClient()
    c.run(500)