# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:40:41 2017

@author: alcor
"""
import scipy.optimize
from scipy.optimize import least_squares
import numpy as np
from raw_utils import read_raw
from image_utils import rmse
from run_simulation import run, potato, vec_str
from image_utils import norm2_diff
from raw_utils import save_png
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import os

os.chdir('./optimization_results')

import pickle

with open('plotdata.pkl', 'rb') as f:
    plotdata = pickle.load(f)

#%%
    
#%%
ref_file = plotdata["ref_name"]
reference = read_raw(ref_file)
save_png('reference.png', reference)

opt = plotdata["opt_name"]

def process(opt, name, res):
    optimum = read_raw(opt)
    save_png(name + '.png', optimum)
    diff = norm2_diff(optimum, reference)
    #%%
    print(diff.max())
    plt.close("all")
    plt.imshow(np.clip(diff,0,0.05), cmap='hot')
    plt.colorbar()
    plt.savefig(name + '_diff.pdf', bbox_inches='tight')
    #%%
    plt.close("all")
    
    data = np.loadtxt(res)
    
    #data = np.array(allvecs)
    ##
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dev = np.std(data[:,3])
    avg = np.mean(data[:,3])
    ax.scatter(data[:,1], data[:,2], data[:,3])
    ax.set_zlim(avg - dev/2, avg + dev/2)
    plt.savefig(name + '_data.pdf', bbox_inches='tight')
    plt.close("all")
    print(data.shape)
    fig = plt.figure()
    plt.plot(data[:,0],data[:,3],'r')
    plt.plot(data[:,1],data[:,3],'g')
    plt.plot(data[:,2],data[:,3],'b')
    plt.savefig(name + '_curves.pdf', bbox_inches='tight')
    plt.close("all")
    
process(opt, 'potato', 'result.txt')