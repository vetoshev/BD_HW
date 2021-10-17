#!/usr/bin/env python
"""reducer.py"""
import sys

ans = [0, 0, 0]

for line in sys.stdin:
    line = line.strip()
    ck, mk, vk = map(float, line.split('\t'))
    c = ans[0] + ck
    m = (ans[0]*ans[1] + ck*mk) / c
    v = (ans[0]*ans[2] + ck*vk) / c + ans[0]*ck*((ans[1] - mk) / c)**2
    ans = [c, m, v]

print(ans)
