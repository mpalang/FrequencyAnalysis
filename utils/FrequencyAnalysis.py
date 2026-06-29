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
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity
import sys
from dataclasses import dataclass

if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))
from utils.auxiliary import get_zero_crossing
from utils.statistic_tools import hist,kde


@dataclass
class Oscillation_Estimate:
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    filter_strength: float = 0.1
    ID: str = 'unknown'
    kernel: str = 'exponential'
    bandwidth: float = 0.1
    
    def __post_init__(self):
        if not isinstance(self.x,np.ndarray) or not isinstance(self.y,np.ndarray):
           raise ValueError('x and y must be numpy arrays') 
        if self.x.shape != self.y.shape:
            raise ValueError('x and y need to be numpy array of same length')
        
        self.validate()
        
        self.dy = np.gradient(self.y,self.x, edge_order=2)
        self.d2y = np.gradient(self.dy,self.x, edge_order=2)
        self.d3y = np.gradient(self.d2y,self.x, edge_order=2)
        self.get_extrema()    
        self.get_diffs()
        self.get_kde()
        self.get_period()
        self.get_phase()
        self.A,self.f,self.phi = self.fit()
        
    
    def validate(self):
        valid_kernels = ['gaussian','epanechnikov','tophat','exponential','linear','cosine']
        if not self.kernel in valid_kernels:
            raise ValueError('kernel must be a string and in this list:\n ["gaussian","epanechnikov","tophat","exponential","linear","cosine"]')
        
        if not isinstance(self.ID,str):
            self.ID = 'unknown'
            raise Warning('ID must be string')
            
        if not isinstance(self.filter_strength,float):
            self.filter_strength = 0.1
            raise Warning('filter_strength must be float')
            
            
    def get_extrema(self):
        condition_Maxima = (self.dy[:-1] > 0) & (self.dy[1:] < 0)
        # Get indices where the condition is True
        indices_Maxima = np.where(condition_Maxima)[0]
        Maxima = []
        for i in indices_Maxima:
            x0 = get_zero_crossing(i,self.x,self.dy)
            Maxima.append(x0)
      
        condition_Minima = (self.dy[:-1] < 0) & (self.dy[1:] > 0)
        # Get indices where the condition is True
        indices_Minima = np.where(condition_Minima)[0]
        Minima = []
        for i in indices_Minima:
            x0 = get_zero_crossing(i,self.x,self.dy)
            Minima.append(x0)
        
        self.minima = np.asarray(Minima)
        self.maxima = np.asarray(Maxima)
        self.extrema = np.sort(np.concat((self.maxima,self.minima))) 
        
        return self.minima,self.maxima,self.extrema

    
    def get_diffs(self,mode = 'Extrema'):
        if mode.lower() == 'extrema':
            self.diffs = np.diff(self.extrema)
        if mode.lower() == 'maxima':
            self.diffs = np.diff(self.maxima)
        if mode.lower() == 'minima':
            self.diffs = np.diff(self.minima)
    
    def get_kde(self):
        self.p,self.kde = kde(self.diffs,bandwidth=self.bandwidth,kernel=self.kernel)

            
    def get_period(self):
        #Apply limits
        # #define limits with available x datapoints
        # T_limit_low = 4*self.filter_strength*min(np.diff(self.x))
        # T_limit_high = abs(self.x[-1]-self.x[0])/2 #at least two periods in dataset
        # T_limit_low_index = np.argmax(T>T_limit_low)
        # T_limit_high_index = np.argmax(T>T_limit_high)
        # if T_limit_high_index==0:
        #     T_limit_high_index = None
        
        # T = T[T_limit_low_index:T_limit_high_index]
        
        self.period_estimate = 2*self.p[np.argmax(self.kde)] 
        self.f_estimate = 1/self.period_estimate
        
        # if T_limit_low>Estimated_Period:
        #     print('ERROR')
        #     exit
        

    def get_phase(self):
            
        mod = self.maxima%self.period_estimate
        
        kde = gaussian_kde(mod)
        x = np.linspace(min(mod),max(mod),1000)
        
        self.phase_shift_estimate = 2*np.pi*x[np.argmax(kde(x))]/self.period_estimate+np.pi/2


    def fit(self):
        def fit_fun(x,A,T,phi):
            return A*np.sin(2*np.pi*x/T+phi)
        
        parms = (1,self.period_estimate,self.phase_estimate)
        try:
            parms,cov = curve_fit(fit_fun,self.x,self.dy,p0=parms)
        except:
            print(f'failed to fit {self.ID}')
        
        self.A,self.T,self.phi = *parms
        self.f = 1/self.T