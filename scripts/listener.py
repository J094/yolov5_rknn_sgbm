#!/usr/bin/env python
# J094
"""
Listener for distance
"""

from utils.rosutils import rosListener


def listen():
    sub = rosListener()
    sub.sub_dist()


if __name__ == '__main__':
    listen()
