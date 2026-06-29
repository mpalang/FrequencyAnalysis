# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:47:32 2025

@author: Mo
"""

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
# from scipy import fft        
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit, minimize
from sklearn.neighbors import KernelDensity
import time
import sys

def progress_bar(current, total, bar_length=50):
    percent = current / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '-' * (bar_length - filled)
    sys.stdout.write(f'\rProgress {current}/{total}: |{bar}| {percent*100:.1f}%')
    sys.stdout.flush()
    
def get_histogram(data,normalized=False,bins='auto'):
    # Histogram
    hist, bins = np.histogram(data, bins=bins)
    if normalized:
        hist = hist / hist.max()
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers,hist


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


def load_data():
    z = np.genfromtxt(Path(Path(__file__).parent,'test_data.csv'),delimiter=',')
    wl = np.genfromtxt(Path(Path(__file__).parent,'test_data_wl.csv'),delimiter=',')
    t = np.genfromtxt(Path(Path(__file__).parent,'test_data_t.csv'),delimiter=',')
    
    
    # self.N = 5000
    # self.t_int=np.linspace(min(self.t),max(self.t),self.N)
    return t,wl,z

t,wl,TA = load_data()

class Oscillation_Estimate:
    def __init__(self,x,y,filter_strength=0.1,ID='Unknown'):
        self.x = x
        self.y = y
        self.filter_strength = filter_strength
        self.ID = ID
        # self.dy_dx = self.grad(y,x)
        # self.d2y_d2x = self.grad(self.dy_dx,x)
            
        # self.get_extrema()        
        
        # self.get_kde()
        
    
    def grad(self,y,x):
        return np.gradient(y,x, edge_order=2)
    
    def get_extrema(self):
        condition_Maxima = (self.dy_dx[:-1] > 0) & (self.dy_dx[1:] < 0)
        # Get indices where the condition is True
        indices_Maxima = np.where(condition_Maxima)[0]
        Maxima = []
        for i in indices_Maxima:
            x0 = get_zero_crossing(i,self.x,self.dy_dx)
            Maxima.append(x0)
      
        condition_Minima = (self.dy_dx[:-1] < 0) & (self.dy_dx[1:] > 0)
        # Get indices where the condition is True
        indices_Minima = np.where(condition_Minima)[0]
        Minima = []
        for i in indices_Minima:
            x0 = get_zero_crossing(i,self.x,self.dy_dx)
            Minima.append(x0)
        
        self.minima = np.asarray(Minima)
        self.maxima = np.asarray(Maxima)
        self.extrema = np.sort(np.concat((self.maxima,self.minima))) 
        
        return self.minima,self.maxima,self.extrema

    
    def get_diffs(self,mode = 'Extrema'):
        if mode.lower() == 'extrema':
            self.diffs = np.diff(self.extrema)
    
    def get_kde(self):
        kde = KernelDensity(
            kernel=self.kernel,
            bandwidth=self.bandwith).fit(self.diffs.reshape(-1,1))
        kde = np.exp(kde.score_samples(P.reshape(-1,1)))
        kde = kde/max(kde)
        
        self.kde = kde
        return kde
            
            
    def get_period(self):
        extrema = self.Extrema
        grad = np.diff(extrema)     
        kde = gaussian_kde(grad)
        T = np.linspace(min(grad),max(grad),1000)
        
        #Apply limits
        #define limits with available x datapoints
        T_limit_low = 4*self.filter_strength*min(np.diff(self.x))
        T_limit_high = abs(self.x[-1]-self.x[0])/2 #at least two periods in dataset
        T_limit_low_index = np.argmax(T>T_limit_low)
        T_limit_high_index = np.argmax(T>T_limit_high)
        if T_limit_high_index==0:
            T_limit_high_index = None
        
        T = T[T_limit_low_index:T_limit_high_index]
        
        Estimated_Period = 2*T[np.argmax(kde(T))] 
        
        if T_limit_low>Estimated_Period:
            print('ERROR')
            exit
        
        self.period = Estimated_Period
        return Estimated_Period
        

    def get_Phase(self):
            
        mod = self.Maxima%self.period
        
        kde = gaussian_kde(mod)
        x = np.linspace(min(mod),max(mod),1000)
        
        phase_shift = 2*np.pi*x[np.argmax(kde(x))]/self.period+np.pi/2
        
        return phase_shift 

    def fit(self):
        def fit_fun(x,A,T,phi):
            return A*np.sin(2*np.pi*x/T+phi)
        
        try:
            parms,cov = curve_fit(fit_fun,self.x,self.dy_dx,p0=(1,self.period,self.phase))    
            _,period,phase = parms
        except:
            print(f'failed to fit {self.ID}')
        
        return period, phase



wl1 = 450
wl2 = 450.5

# TA = TA[np.argmax(wl>wl1):np.argmax(wl>wl2),:]
# wl = wl[np.argmax(wl>wl1):np.argmax(wl>wl2)]

Period = []
Phase = []
Maxima = []
Minima = []
Extrema = []


for n,i in enumerate(wl):
    # print(i)
    OE = Oscillation_Estimate(t,TA[n,:],filter_strength=0,ID=str(round(i)))
    Period.append(OE.period)
    Phase.append(OE.phase)
    Maxima.append(OE.Maxima)
    Minima.append(OE.Minima)    
    Extrema.append(OE.Extrema)
    progress_bar(n + 1, len(wl))
    
    
#%%

cmap = plt.get_cmap("seismic")

# fig,ax = plt.subplots()
fig2,ax2 = plt.subplots()
fig3,ax3 = plt.subplots()

kernels = ['gaussian','epanechnikov','tophat','exponential','linear','cosine']
# kernels = ['exponential']
# kernels = ['gaussian']
bandwidths = np.linspace(0.18,0.2,10)
Xs = wl[::100]

N = len(Xs)*len(bandwidths)*len(kernels)

for m,k in enumerate(kernels):
  for o,b in enumerate(bandwidths):
    
    Results = []
    for n,l in enumerate(Xs):
        
        nn = m*len(bandwidths)*len(Xs)+o*len(Xs)+n
        i = int(np.where(wl==l)[0][0])
        color = cmap(nn / N)
        progress_bar(nn + 1, N)
        
        Diffs = np.diff(Extrema[i])
        # Diffs = np.diff(Maxima[i])
        # Diffs = np.diff(Minima[i])
        P = np.linspace(min(Diffs),max(Diffs),100)
    
        # ax.plot(t,TA[i,:])
        # sns.violinplot(x=Diffs,ax=ax)
        # sns.swarmplot(x=Diffs,ax=ax)
        
        # Ps,hist = get_histogram(Diffs,bins=20,normalized=True)
        # ax2.bar(Ps,hist)
        
        #kernel-options: 'gaussian','epanechnikov','tophat','exponential','linear	','cosine'
        
        kde = KernelDensity(
            kernel=k,
            bandwidth=b).fit(Diffs.reshape(-1,1))
        kde = np.exp(kde.score_samples(P.reshape(-1,1)))
        kde = kde/max(kde)
        # ax2.plot(P,kde,color=color)
        
        Pmin, Pmax = 2, 12
        mask = (P >= Pmin) & (P <= Pmax)
        Results.append(P[np.argmax(kde[mask])])
    
    ax3.plot(Xs,Results,label=f'{k}@{b}')
    
ax3.legend(loc="center left", bbox_to_anchor=(1, 0.5))





    
