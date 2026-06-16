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

def Oscillation_Estimate(x,y,x0=0,filter_strength=1,ID='Unknown'):
    
    y = y[np.argmax(x>x0):]
    x = x[np.argmax(x>x0):]
    
    dy_dx = np.gradient(y,x) 

    def get_extrema(x,y):
        """
        """    

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
        # ax.axvline(x=T_limit_low,color='red',linewidth=1.5)
        # # ax.set_xlim(0,10)
        
        #Apply limits
        #define limits with available x datapoints
        T_limit_low = 4*filter_strength*min(np.diff(x))
        T_limit_high = abs(x[-1]-x[0])/2 #at least two periods in dataset
        T_limit_low_index = np.argmax(T>T_limit_low)
        T_limit_high_index = np.argmax(T>T_limit_high)
        if T_limit_high_index==0:
            T_limit_high_index = None
        
        T = T[T_limit_low_index:T_limit_high_index]
        
        Estimated_Period = 2*T[np.argmax(kde(T))] 
        
        # print(f'Estimated Period: {2*T[np.argmax(kde(T))]}')
        
        if T_limit_low>Estimated_Period:
            print('ERROR')
            exit
            
        return Estimated_Period
    
    period = get_period(Extrema) 
    
    def get_Phase(Maxima,Minima,period):
            
        mod = Maxima%period
        
        kde = gaussian_kde(mod)
        x = np.linspace(min(mod),max(mod),1000)
        
        # fig,ax = plt.subplots()
        # ax.scatter(mod,[0]*len(Maxima))
        # ax.plot(x,kde(x))
        
        phase_shift = 2*np.pi*x[np.argmax(kde(x))]/period+np.pi/2
        
        # print(f'''
# Estimates for
# Period: {round(period,2)}
# Phase Shift: {round(phase_shift/np.pi,2)} \u03C0
#               ''')
        
        return phase_shift 
        
    phase = get_Phase(Maxima,Minima,period)
        

    def fit_fun(x,A,T,phi):
        return A*np.sin(2*np.pi*x/T+phi)
    
    try:
        parms,cov = curve_fit(fit_fun,x,dy_dx,p0=(1,period,phase))    
        _,period,phase = parms
    except:
        print(f'failed to fit {ID}')

    
    # fig,ax=plt.subplots(2,1)
    # ax[0].plot(x,y)
    # ax[1].plot(x,dy_dx)  
    # ax[1].plot(Maxima,np.array([0]*len(Maxima)),'x')
    # ax[1].plot(Minima,np.array([0]*len(Minima)),'x')
    # # ax[1].plot(Extrema,np.array([0]*len(Extrema)),'x')
    # ax[1].plot(x, fit_fun(x,*parms),':')
    # # ax[0].set_xlim([45,65])
    # ax[1].set_ylim([-0.07,0.07])
    # # ax[1].set_xlim([45,65])
    # ax[1].axhline(color='k')

    
    return period, phase

t = FA.t
wl = FA.wl
TA = FA.exp_data

wl1 = 450
wl2 = 450.5

# TA = TA[np.argmax(wl>wl1):np.argmax(wl>wl2),:]
# wl = wl[np.argmax(wl>wl1):np.argmax(wl>wl2)]

Period = np.zeros(TA.shape[0])
Phase = np.zeros(TA.shape[0])

#%

start = time.perf_counter()

for n,i in enumerate(wl):
    Period[n],Phase[n] = Oscillation_Estimate(t,TA[n,:],filter_strength=0.0001,ID=str(i))

end = time.perf_counter()

print(f"Elapsed time: {end - start:.2f} seconds")


fig,ax = plt.subplots(2,1)
ax[0].axhline(0,color='k')
ax[0].plot(wl,2*np.pi/Period)
# ax[0].plot(wl,Period)
ax[1].plot(wl,Phase)
ax[0].set_ylim((0,2))


#%%

fig,ax=plt.subplots()
cs = ax.contourf(wl,t,TA.T,levels=30)
fig.colorbar(cs)
