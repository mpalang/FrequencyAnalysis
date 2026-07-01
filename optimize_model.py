# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:05:24 2026

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

import optuna

data = load_data(xlim=(4,115))
t,wl,TA = data.x,data.y,data.z
t_int = data.x_int
wl_ = wl[::200]


def objective(trial):

    fit_space_phi = trial.suggest_float("fs_phi", 0, 0.1)
    fit_space_A = trial.suggest_float("fs_A", 0, 5)
    fit_space_f = trial.suggest_float("fs_f", 0, 5)
    # use_f = trial.suggest_categorical("use_f",[True,False])
    use_est = trial.suggest_categorical("use_est",[True,False])
    kernel = trial.suggest_categorical("kernel",
['gaussian','epanechnikov','tophat','exponential','linear','cosine'])
    bandwidth = trial.suggest_float("bandwidth",0,5)
    
    O = []
    t0 = pc()
    for m,i in enumerate(wl_):
        # pb(m,len(wl_))
        n = np.argmax(wl>i)
        O.append(Oscillation_Analyser(t,TA[:,n],use_estimates=use_est,
                                     fit_space=(fit_space_A,fit_space_f,fit_space_phi),
                                     kernel = kernel,bandwidth=bandwidth,
                                     ))
    
    time_sensitivity = 0
    loss = sum([o.error for o in O])/len(O)
    loss = loss + (pc()-t0)*time_sensitivity
    
    return loss

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize')
study.optimize(objective,n_trials=10)

print(f"""
{study.best_params}
{study.best_value}
      """)