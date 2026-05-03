# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:47:32 2025

@author: Mo
"""

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from scipy import fft


class FrequencyAnalysis:
    def __init__(self):
        self.load_data()
        
        
        # self.make_model()
        
        # self.A,self.f, self.phi = self.predict_params()


# =============================================================================
#     #%functions
# =============================================================================
    def load_data(self):
        self.exp_data = np.genfromtxt(Path(Path(__file__).parent,'test_data.csv'),delimiter=',')
        self.wl = np.genfromtxt(Path(Path(__file__).parent,'test_data_wl.csv'),delimiter=',')
        self.t = np.genfromtxt(Path(Path(__file__).parent,'test_data_t.csv'),delimiter=',')
        
        self.N = 5000
        self.t_int=np.linspace(min(self.t),max(self.t),self.N)
      

    
 


# =============================================================================
# Execute:
# =============================================================================

FA = FrequencyAnalysis()


#%% Diff Test

t = FA.t
t_int = np.linspace(min(t),max(t),5000)
y = FA.exp_data[1000,:]
y_int = np.interp(t_int,t,y)

y_diff = np.diff(y_int)   

fig,ax=plt.subplots(2,1)
ax[0].plot(t,y)
ax[0].plot(t_int,y_int)
ax[1].plot(t_int[:-1],y_diff)





def get_Extrema():
    
    # Find zero slope (maxima & minima)
    condition_Maxima = (y_diff[:-1] > 0) & (y_diff[1:] < 0)
    # Get indices where the condition is True
    indices_Maxima = np.where(condition_Maxima)[0]
    
    condition_Minima = (y_diff[:-1] < 0) & (y_diff[1:] > 0)
    
    # Get indices where the condition is True
    indices_Minima = np.where(condition_Minima)[0]
    
    Minima = t_int[indices_Minima]
    Maxima = t_int[indices_Maxima]
    
    def pick(extrema):
        selected = []
        for i in range(1, len(extrema)-1):
            prev_el = extrema[i-1]
            el = extrema[i]
            next_el = extrema[i+1]
            
            if 1.2 > abs(el - prev_el)/abs(el - next_el) > 0.8:
                selected.append(el)
            
        return np.asarray(selected)
    
    Minima = pick(Minima)
    Maxima = pick(Maxima)
    
    return Minima, Maxima


Minima,Maxima = get_Extrema()  
    
ax[1].plot(Maxima,np.array([0.003]*len(Maxima)),'x')
ax[1].plot(Minima,np.array([-0.003]*len(Minima)),'x')

ax[1].plot(t_int, 0.001*np.sin(2*np.pi*(t_int+6)/11.4))

ax[1].set_ylim([-0.005,0.005])
ax[1].axhline(color='k')

#%%

Tn = np.median(np.diff(Minima[:]))
Tp = np.median(np.diff(Maxima[:]))

print(Tn)
print(Tp)









