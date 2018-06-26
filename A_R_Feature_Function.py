# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:23:17 2016

@author: shengdi
"""

import essentia

# as there are 2 operating modes in essentia which have the same algorithms,
# these latter are dispatched into 2 submodules:

from essentia.standard import *
import essentia.streaming

from pylab import plot, show, figure
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.signal import find_peaks_cwt, butter, lfilter, lp2hp
import peakutils
#from peakdetect import peakdetect

from sklearn import linear_model
from pandas import DataFrame
from operator import itemgetter

import sys
sys.path.append('/Users/shengdi/Documents/tools/SuMPF/source/_sumpf/_modules/_interpretations')
import sumpf
from signalenvelope import SignalEnvelope
  
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



def attack_feature(audio, threshold):   
    '''
    loader = essentia.standard.MonoLoader(filename =
            '/Users/shengdi/Documents/code/database/violin/violin1-Attack10.wav')
    Audio = loader()
    audio = EL(Audio)
    '''
    signal = np.reshape(audio,[1,audio.size])
    signal = sumpf.Signal(signal,44100.0)
    
    obj = SignalEnvelope(signal)
    a = obj.GetEnvelope()
    b = np.array(a.GetChannels())
    b = np.reshape(b,b.size)
    
    #highpass - smooth, doesn't seem to work
    '''
    order = 6
    fs = 30.0       # sample rate, Hz
    cutoff = 2.0  # desired cutoff frequency of the filter, Hz
    x = butter_highpass_filter(b, cutoff, fs, order)
    '''
    
    #plt.plot(sp.fft(x))
    y_top = peakutils.indexes(b, thres=threshold, min_dist=100)
    y_bottom = peakutils.indexes(threshold-b, thres=0, min_dist=100)
    
    top_label = np.array([y_top,np.zeros(y_top.size)])
    bottom_label = np.array([y_bottom,np.ones(y_bottom.size)])
    
    labels = np.concatenate((top_label,bottom_label), axis=1)
    sorted_labels = np.array(sorted(np.transpose(labels),key=itemgetter(0)))
    
    #interval = []
    #start_point = []
    slop = []
    for i in range(1,sorted_labels.size/2):
        j = i-1
        if sorted_labels[i,1]==1:
            i+=1
        else:
            while(sorted_labels[j,0]==0):
                j-=1
            k = sorted_labels[i,0]-sorted_labels[j,0]
            #interval.append(k)
            #start_point.append(i)
            slop.append((b[sorted_labels[i,0]]-b[sorted_labels[j,0]])/k)
        
    
    attack_feature_AVR = np.mean(slop)
    attack_feature_VAR = np.var(slop)
    
    return attack_feature_AVR, attack_feature_VAR
    

def release_feature(audio, threshold):   
    '''
    loader = essentia.standard.MonoLoader(filename =
            '/Users/shengdi/Documents/code/database/violin/violin1-Release300.wav')
    Audio = loader()
    audio = EL(Audio)
    '''
    signal = np.reshape(audio,[1,audio.size])
    signal = sumpf.Signal(signal,44100.0)
    
    obj = SignalEnvelope(signal)
    a = obj.GetEnvelope()
    b = np.array(a.GetChannels())
    b = np.reshape(b,b.size)
    
    #highpass - smooth, doesn't seem to work
    '''
    order = 6
    fs = 30.0       # sample rate, Hz
    cutoff = 2.0  # desired cutoff frequency of the filter, Hz
    x = butter_highpass_filter(b, cutoff, fs, order)
    '''
    
    #plt.plot(sp.fft(x))
    y_top = peakutils.indexes(b, thres=threshold, min_dist=100)
    y_bottom = peakutils.indexes(threshold-b, thres=0, min_dist=100)
    
    top_label = np.array([y_top,np.zeros(y_top.size)])
    bottom_label = np.array([y_bottom,np.ones(y_bottom.size)])
    
    labels = np.concatenate((top_label,bottom_label), axis=1)
    sorted_labels = np.array(sorted(np.transpose(labels),key=itemgetter(0)))
    
    #interval = []
    #start_point = []
    slop = []
    for i in range(1,sorted_labels.size/2):
        j = i-1
        if sorted_labels[i,1]==0:
            i+=1
        else:
            while(sorted_labels[j,0]==1):
                j-=1
            k = sorted_labels[i,0]-sorted_labels[j,0]
            #interval.append(k)
            #start_point.append(i)
            slop.append((b[sorted_labels[j,0]]-b[sorted_labels[i,0]])/k)
        
    
    release_feature_AVR = np.mean(slop)
    release_feature_VAR = np.var(slop)
    
    return release_feature_AVR, release_feature_VAR
    
'''
def attack_feature_7th(audio, threshold): 

    loader = essentia.standard.MonoLoader(filename =
            '/Users/shengdi/Documents/code/database/violin/violin1-Attack10.wav')
    Audio = loader()
    audio = EL(Audio)


    frame_size = 32
    frame_num = Audio.size/frame_size
    rms = np.zeros(frame_num)
    for i in range(frame_num):
        rms[i] = sum(pow(Audio[i*frame_size:i*frame_size+512],2))/frame_size
        
    rms_curve = butter_lowpass_filter(rms,10,400)
    
    peaks = peakutils.indexes(rms_curve, thres=threshold, min_dist=300)[0]
    
    return rms_curve[peaks]
'''