# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 14:47:32 2025

@author: Mo
"""

import numpy as np
from pathlib import Path
from time import perf_counter as pc

# from scipy import fft        
from scipy.optimize import curve_fit,least_squares

import sys
from dataclasses import dataclass


if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))
from utils.auxiliary import get_zero_crossing
from utils.statistic_tools import hist,kde,kde_argmax


@dataclass
class Oscillation_Analyser:
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    filter_strength: float = 0.1
    ID: str = 'unknown'
    kernel: str = 'exponential'
    bandwidth: float = 3
    use_estimates: bool = False
    use_frequency: bool = False
    fit_space: tuple = (3,3,0.1)
    
    def __post_init__(self):

        self.validate()
        
        self.dy = np.gradient(self.y,self.x, edge_order=2)
        self.d2y = np.gradient(self.dy,self.x, edge_order=2)
        self.d3y = np.gradient(self.d2y,self.x, edge_order=2)

        
        t0=pc()
        self.get_extrema()    
        self.get_diffs()
        self.get_period()
        self.get_phase()
        self.get_amplitude()
        t1=pc()
        self.get_fitfun()
        if self.use_estimates:
            self.return_estimates()
        else:
            self.fit()
        t2=pc()
        

        self.t1 = t1-t0
        self.t2 = t2-t1
        
    
    def validate(self):
        if not isinstance(self.x,np.ndarray) or not isinstance(self.y,np.ndarray):
           raise ValueError('x and y must be numpy arrays') 
        
        if self.x.shape != self.y.shape:
            raise ValueError('x and y need to be numpy array of same length')
            
        valid_kernels = ['gaussian','epanechnikov','tophat','exponential','linear','cosine']
        if not self.kernel in valid_kernels:
            raise ValueError('kernel must be a string and in this list:\n ["gaussian","epanechnikov","tophat","exponential","linear","cosine"]')
        
        if not isinstance(self.ID,str):
            self.ID = 'unknown'
            raise Warning('ID must be string')
            
        if not isinstance(self.filter_strength,float):
            self.filter_strength = 0.1
            raise Warning('filter_strength must be float')
            
        if not isinstance(self.fit_space,tuple):
            self.fit_space = (0.1,0.1,0.1)
            raise Warning('fit_space must be tuple of floats and in [0,1]')
            
            
    def get_extrema(self):
        condition_Maxima = (self.d2y[:-1] > 0) & (self.d2y[1:] < 0)
        # Get indices where the condition is True
        indices_Maxima = np.where(condition_Maxima)[0]
        Maxima = []
        for i in indices_Maxima:
            x0 = get_zero_crossing(i,self.x,self.d2y)
            Maxima.append(x0)
      
        condition_Minima = (self.d2y[:-1] < 0) & (self.d2y[1:] > 0)
        # Get indices where the condition is True
        indices_Minima = np.where(condition_Minima)[0]
        Minima = []
        for i in indices_Minima:
            x0 = get_zero_crossing(i,self.x,self.d2y)
            Minima.append(x0)
        
        self.minima = np.asarray(Minima)
        self.i_minima = indices_Minima
        self.maxima = np.asarray(Maxima)
        self.i_maxima = indices_Maxima
        self.extrema = np.sort(np.concat((self.maxima,self.minima))) 
        
        return self.minima,self.maxima,self.extrema

    
    def get_diffs(self,mode = 'Extrema'):
        if mode.lower() == 'extrema':
            self.diffs = np.diff(self.extrema)
        if mode.lower() == 'maxima':
            self.diffs = np.diff(self.maxima)
        if mode.lower() == 'minima':
            self.diffs = np.diff(self.minima)
            
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
        
        self.T_estimate = 2*kde_argmax(self.diffs,bandwidth=self.bandwidth,kernel=self.kernel)
        self.f_estimate = 1/self.T_estimate
        
        # if T_limit_low>Estimated_Period:
        #     print('ERROR')
        #     exit
        
    def get_phase(self):
        modmax = self.maxima%self.T_estimate #project onto T_estimate interval
        DTmax = self.T_estimate - kde_argmax(modmax,bandwidth=self.bandwidth,kernel=self.kernel)
        pmax = (2*np.pi*DTmax)/self.T_estimate +0.5*np.pi # 0.5pi accounts for first sin maximum not at zero
        modmin = self.minima%self.T_estimate #project onto T_estimate interval
        DTmin = self.T_estimate - kde_argmax(modmin,bandwidth=self.bandwidth,kernel=self.kernel)
        pmin = (2*np.pi*DTmin)/self.T_estimate -0.5*np.pi # 1.5pi accounts for first sin minimum not at zero
        self.phase_estimate = (pmax+pmin)/2
        
        return self.phase_estimate

    
    def get_amplitude(self):
        
        Amin = np.abs(np.median(self.dy[self.i_minima]))
        Amax = np.abs(np.median(self.dy[self.i_maxima]))
        
        self.A_estimate = (Amin+Amax)/2

    def return_estimates(self):
        self.A = self.A_estimate
        self.f = self.f_estimate
        self.T = self.T_estimate
        self.phi = self.phase_estimate
        self.error = np.sum((self.fit_fun(self.x,self.A,self.T,self.phi)-self.dy)**2)
    
    def get_fitfun(self):
        if self.use_frequency:
            def fit_fun(x,A,f,phi):
                return A*np.sin(2*np.pi*f*x+phi)

            self.fit_fun = fit_fun
            self.x0 = [self.A_estimate, self.f_estimate, self.phase_estimate]
            self.xlow = [self.A_estimate-self.A_estimate*self.fit_space[0],
                         self.f_estimate-self.f_estimate*self.fit_space[1],
                         self.phase_estimate-self.phase_estimate*self.fit_space[2],]
            self.xhigh = [self.A_estimate+self.A_estimate*self.fit_space[0],
                         self.f_estimate+self.f_estimate*self.fit_space[1],
                         self.phase_estimate+self.phase_estimate*self.fit_space[2],]
        else:
            def fit_fun(x,A,T,phi):
                return A*np.sin(2*np.pi*x/T+phi)
            self.fit_fun = fit_fun
            self.x0 = [self.A_estimate, self.T_estimate, self.phase_estimate]
            self.xlow = [self.A_estimate-self.A_estimate*self.fit_space[0],
                         self.T_estimate-self.T_estimate*self.fit_space[1],
                         self.phase_estimate-self.phase_estimate*self.fit_space[2],]
            self.xhigh = [self.A_estimate+self.A_estimate*self.fit_space[0],
                         self.T_estimate+self.T_estimate*self.fit_space[1],
                         self.phase_estimate+self.phase_estimate*self.fit_space[2],]
        
    
    def fit(self):
        def residuals(params, x, dy):
            A, Tf, phi = params
            res1 = self.fit_fun(x, A, Tf, phi) - self.dy
            # A2 = 2*np.pi*Tf*A
            # res2 = self.fit_fun(x,A2,Tf,phi+np.pi/2) - self.d2y
            # A3 = 2*np.pi*Tf*2
            # res3 = self.fit_fun(x,A3,Tf,phi+np.pi) - self.d3y
            
            return res1
        
        # try:
        result = least_squares(
                                residuals,
                                x0= self.x0,
                                bounds=(self.xlow,self.xhigh),
                                args=(self.x,self.dy)
                                )
        parms = result.x
        error = result.cost*2
        
        if self.use_frequency:
            self.A,self.f,self.phi = parms
            self.T = 1/self.f
        else:
            self.A,self.T,self.phi =parms
            self.f = 1/self.T
        self.error = error