# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:02:16 2026

@author: morit
"""

import numpy as np
from pathlib import Path

class load_data:
    def __init__(self,xlim=(None,None),ylim=(None,None),n_int=1000):
        z = np.genfromtxt(Path(Path(__file__).parent.parent,'TestData','test_data.csv'),delimiter=',').T
        y = np.genfromtxt(Path(Path(__file__).parent.parent,'TestData','test_data_wl.csv'),delimiter=',')
        x = np.genfromtxt(Path(Path(__file__).parent.parent,'TestData','test_data_t.csv'),delimiter=',')
        
        ix1 = iy1 = 0
        ix2 = len(x)
        iy2 = len(y)
        if xlim[0]:
            ix1 = np.argmax(x>xlim[0])
        if xlim[1]:
            ix2 = np.argmax(x>xlim[1])
        if ylim[0]:
            iy1 = np.argmax(y>ylim[0])
        if ylim[1]:
            iy1 = np.argmax(y>ylim[0])
            
            
        self.x = x[ix1:ix2]
        self.y= y[iy1:iy2]
        self.z = z[ix1:ix2,iy1:iy2]

        self.x_int=np.linspace(min(self.x),max(self.x),n_int)