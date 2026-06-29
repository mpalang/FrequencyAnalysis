# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:00:57 2026

@author: moritzpalang
"""
from pathlib import Path
import sys

if str(Path(__file__).parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent))
from utils.io import load_data
from utils.FrequencyAnalysis import Oscillation_Estimate
from utils.statistic_tools import hist,kde
from utils.auxiliary import progress_bar as pb
from matplotlib import pyplot as plt
import numpy as np

data = load_data(xlim=(3,115))
t,wl,TA = data.x,data.y,data.z
t_int = data.x_int

Fs=[]
for n,i in enumerate(wl[::100]):
    pb(n,len(wl))
    oe = Oscillation_Estimate(t,TA[:,n])
    Fs.append(oe.f)


#%%
# fig,ax = plt.subplots()

# xbin,hist = hist(oe.diffs,norm=True,bins=10)
# ax.bar(xbin,hist)
# ps,ks = kde(oe.diffs,bandwidth=0.2)
# ax.plot(ps,ks,color='red')

# fig,ax = plt.subplots()
# ax.axhline(0,color='k',linewidth=1)
# ax.plot(t,oe.dy,label='d1')
# ax.plot(t_int,np.sin(2*np.pi*t_int/oe.period)*max(oe.dy))
# ax.plot(t,oe.d2y,label='d2')
# ax.plot(t,oe.d3y,label='d3')
# ax.set_ylim((-1,1))
# ax.set_xlim((0,10))
# ax.legend()

fig,ax = plt.subplots()
ax.plot(wl,Fs)
ax.set_ylim(0,0.2)





