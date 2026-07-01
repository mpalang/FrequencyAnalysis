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
from utils.FrequencyAnalysis import Oscillation_Analyser
from utils.statistic_tools import kde,kde_argmax
from utils.auxiliary import progress_bar as pb
from matplotlib import pyplot as plt
import numpy as np
from time import perf_counter as pc

data = load_data(xlim=(4,115))
t,wl,TA = data.x,data.y,data.z
t_int = data.x_int

Oes=[]
Ts=[]
t0 = pc()
wl_ = wl[::1]
for m,i in enumerate(wl_):
    n = int(np.argmax(wl>i))
    t1=pc()
    pb(m,len(wl_))
    Oes.append(Oscillation_Analyser(t,TA[:,n]))
    Ts.append(pc()-t1)
T_total = pc()-t0

print(f"""
Average time: {T_total/len(wl_)}
Relative fit time: {sum([oe.t2 for oe in Oes])/T_total}
Average error: {sum([oe.error for oe in Oes])/len(wl_)}
""")
# Total time: {T_total}
# Sum fit: {sum([oe.t2 for oe in Oes])} ({sum([oe.t2 for oe in Oes])/T_total})
# Sum outer loop: {sum(Ts)} ({sum(Ts)/T_total})
# Sum estimate: {sum([oe.t1 for oe in Oes])} ({sum([oe.t1 for oe in Oes])/T_total})
#%%

fig,axs = plt.subplots(2,2)
fs = [oe.f for oe in Oes]
axs[0,0].plot(wl_,fs)
axs[0,0].set_ylim(0,np.median(fs)*2)
axs[0,0].set_title('Frequency')

phis = [oe.phi for oe in Oes]
axs[1,0].plot(wl_,phis)
# axs[1,0].set_ylim(0,2*np.pi)
axs[1,0].set_title('phase')

As = [oe.A for oe in Oes]
axs[0,1].plot(wl_,As)
# axs[0,1].set_ylim(0,np.median(As)*6)
axs[0,1].set_title('Amplitude')

Es = [oe.error for oe in Oes]
axs[1,1].plot(wl_,Es)
# axs[1,1].set_ylim(0,np.median(Es)*20)
axs[1,1].set_title('error')

plt.tight_layout()

#%% Inspect:

x = 370
idx = np.argmax(wl_>x)
idx = 235
o = Oes[idx]

def get_phase(T_estimate,maxima,minima):
    
    # fig,ax = plt.subplots()
    modmax = maxima%T_estimate #project onto T_estimate interval
    DT = T_estimate - kde_argmax(modmax)
    pmax = (2*np.pi*DT)/T_estimate + 0.5*np.pi # 0.5pi accounts for first sin maximum not at zero

    modmin = minima%T_estimate #project onto T_estimate interval
    DT = T_estimate - kde_argmax(modmin)
    pmin = (2*np.pi*DT)/T_estimate -0.5*np.pi # 0.5pi accounts for first sin maximum not at zero
    
    # ax.plot(*kde(modmax))
    # ax.plot(*kde(modmin))
    
    return pmax,pmin


pmax,pmin = get_phase(o.T,o.maxima,o.minima)
print(pmax)
print(pmin)


#%
fig,ax = plt.subplots()
y = o.dy
yfit = o.fit_fun(t,o.A,o.f,o.phi-0.01*np.pi)
res = y-yfit
ax.plot(t,y,c='k')
ax.plot(t,yfit,c='green')
ax.plot(t,res,':',c='red')
# ax.scatter(o.maxima,np.zeros(o.maxima.shape))
# ax.scatter(o.minima,np.zeros(o.minima.shape))
ax.set_title(str(np.sum(res**2)))

# ytest = o.fit_fun(t,o.A,o.f,phi)
# ax.plot(t,ytest,c='blue')




