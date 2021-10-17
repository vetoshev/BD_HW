#!/usr/bin/env python
"""mapper.py"""
import sys

for line in sys.stdin:
    lines = line.split(',')
    try:
        price = lines[-1]
        if price == '':
            raise ValueError
        print('%s\t%s\t%s' % (1, float(price), 0))
    except ValueError:
        # count was not a number, so silently
        # ignore/discard this line
        continue
