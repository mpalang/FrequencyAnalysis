# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:03:43 2026

@author: morit
"""
import sys

def progress_bar(current, total, bar_length=50):
    percent = (current+1) / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '-' * (bar_length - filled)
    sys.stdout.write(f'\rProgress {current+1}/{total}: |{bar}| {percent*100:.1f}%')
    sys.stdout.flush()
    

def get_zero_crossing(i,x,y):
    """get's the x value of where line between (x[i],y[i]) and (x[i+1],y[i+1]) crosses zero."""
    if i<len(x)-1:
        x1 = x[i]
        x2 = x[i+1]
        y1 = y[i]
        y2 = y[i+1]
        a = (y1-y2)/(x1-x2)
        b = y1 - a*x1
        x0 = -b/a
        return x0
    else:
        x[i]