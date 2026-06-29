# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:03:43 2026

@author: morit
"""
import numpy as np
from sklearn.neighbors import KernelDensity

def hist(data,norm=False,bins='auto'):
    # Histogram
    hist, bins = np.histogram(data, bins=bins)
    if norm:
        hist = hist / hist.max()
    
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers,hist


def kde(data,bandwidth=0.1,kernel='gaussian'):
    x = np.linspace(np.min(data),np.max(data),1000)
    kde = KernelDensity(
        kernel=kernel,
        bandwidth=bandwidth).fit(data.reshape(-1,1))
    kde = np.exp(kde.score_samples(x.reshape(-1,1)))
    kde = kde/max(kde)

    return x,kde