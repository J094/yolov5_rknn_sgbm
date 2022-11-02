#!/usr/bin/env python
# J094
"""
Service test
"""

from utils.rosutils import rosService


if __name__ == '__main__':
    s = rosService()
    s.run()
