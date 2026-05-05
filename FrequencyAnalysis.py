# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:47:32 2025

@author: Mo
"""

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
# from scipy import fft        
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit, minimize


#TODO: See what part of the function takes the longest
#TODO: Fix Phase sign issue (0.5pi instead of 1.5 e.g.)
#TODO: Optimize robustness (lots of nonsensical results still). Maybe discard datapoints that don't make sense? Find criterium to discard datapoints?
 
import time

start = time.perf_counter()
end = time.perf_counter()

print(f"Elapsed time: {end - start:.2f} seconds")






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


#% Diff Test


# Period = 10
# Phase_Shift = 0.5*np.pi


# t = np.linspace(min(t),max(t),5000)
# ta = 0.2*np.sin((2*np.pi*t/Period)-Phase_Shift)

def Oscillation_Estimate(x,y):

    def get_extrema(x,y):
        """
        """    
        dy_dx = np.gradient(y,x) 
        
        condition_Maxima = (dy_dx[:-1] > 0) & (dy_dx[1:] < 0)
        # Get indices where the condition is True
        indices_Maxima = np.where(condition_Maxima)[0]
        Maxima = []
        for i in indices_Maxima:
            x1 = x[i]
            x2 = x[i+1]
            y1 = dy_dx[i]
            y2 = dy_dx[i+1]
            a = (y1-y2)/(x1-x2)
            b = y1 - a*x1
            x0 = -b/a
            Maxima.append(x0)
      
        condition_Minima = (dy_dx[:-1] < 0) & (dy_dx[1:] > 0)
        # Get indices where the condition is True
        indices_Minima = np.where(condition_Minima)[0]
        Minima = []
        for i in indices_Minima:
            x1 = x[i]
            x2 = x[i+1]
            y1 = dy_dx[i]
            y2 = dy_dx[i+1]
            a = (y1-y2)/(x1-x2)
            b = y1 - a*x1
            x0 = -b/a
            Minima.append(x0)
        
        # fig,ax=plt.subplots(2,1)
        # ax[0].plot(t,ta)
        # # ax[0].plot(t_int,y_int)
        # ax[1].plot(t,dy_dx)  
        # ax[1].plot(Maxima,np.array([0]*len(Maxima)),'x')
        # ax[1].plot(Minima,np.array([0]*len(Minima)),'x')
        # # ax[1].plot(Extrema,np.array([0]*len(Extrema)),'x')
        # # ax[1].plot(t_int, 0.001*np.sin(2*np.pi*(t_int+6)/11.4))
        # # ax[0].set_xlim([45,65])
        # ax[1].set_ylim([-0.07,0.07])
        # # ax[1].set_xlim([45,65])
        # ax[1].axhline(color='k')
        
        return np.asarray(Maxima),np.asarray(Minima)
    
    Maxima,Minima = get_extrema(x,y)
    Extrema = np.sort(np.concat((Maxima,Minima))) 

    def get_period(extrema):
        grad = np.diff(extrema)     
        kde = gaussian_kde(grad)
        T = np.linspace(min(grad),max(grad),1000)
        
        # fig,ax = plt.subplots()
        # ax.plot(T,kde(T))
        # ax.plot(grad,[0.08]*len(grad),'x')
        
        # print(f'Estimated Period: {2*T[np.argmax(kde(T))]}')
        return 2*T[np.argmax(kde(T))] 
    
    period = get_period(Extrema) 
    
    def get_Phase(Maxima,period):
            
        mod = Maxima%period
        
        kde = gaussian_kde(mod)
        x = np.linspace(min(mod),max(mod),1000)
        
        # fig,ax = plt.subplots()
        # ax.scatter(mod,[0]*len(Maxima))
        # ax.plot(x,kde(x))
        
        phase_shift = 2*np.pi*x[np.argmax(kde(x))]/period
        
        # print(f'''
# Estimates for
# Period: {round(period,2)}
# Phase Shift: {round(phase_shift/np.pi,2)} \u03C0
#               ''')
        
        return phase_shift 
        
    phase = get_Phase(Maxima,period)
    
    return period, phase

cut_early = 0
t = FA.t[cut_early:]
wl = FA.wl
TA = FA.exp_data[:,cut_early:]

Period = np.zeros(TA.shape[0])
Phase = np.zeros(TA.shape[0])
for n,i in enumerate(wl):
    Period[n],Phase[n] = Oscillation_Estimate(t,TA[n,:])
    
fig,ax = plt.subplots(2,1)
ax[0].axhline(0,color='k')
ax[0].plot(wl,2*np.pi/Period)
ax[1].plot(wl,Phase)
ax[0].set_ylim((0,2))
