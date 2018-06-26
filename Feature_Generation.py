# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:10:27 2017

@author: shengdi
"""


import essentia.standard
import essentia.streaming

from pylab import plot, show, figure
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks_cwt, butter, lfilter, lp2hp

from sklearn import linear_model
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor
from pandas import DataFrame

from essentia.standard import *

from A_R_Feature_Function import attack_feature, release_feature

import random
from tempfile import TemporaryFile
from scipy import interpolate

import sys
sys.path.append('/Users/shengdi/Documents/tools/SuMPF/source/_sumpf/_modules/_interpretations')
import sumpf
from signalenvelope import SignalEnvelope
import peakutils
from operator import itemgetter

#import madmom
#proc = madmom.features.onsets.CNNOnsetProcessor()
from essentia import Pool, array



w = Windowing(type = 'hann')    
Loud = Loudness()
EL = EqualLoudness()
spectrum = Spectrum()
mfcc = MFCC()
centrol_moment = CentralMoments()
distribution_shape = DistributionShape()
centroid = Centroid()
rms = RMS()
low_pass = LowPass(cutoffFrequency=500)
high_pass = HighPass(cutoffFrequency=2000)


od = OnsetDetection(method = 'hfc')
#od = OnsetDetection(method = 'flux')
onsets = Onsets()
fft = FFT() # this gives us a complex FFT
c2p = CartesianToPolar() # and this turns it into a pair (magnitude, phase)

pool = Pool()




def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
    

def Feature_Frequency(Audio):  
    s = Audio.size/512+10
    f1 = np.zeros(s)
    f2 = np.zeros(s)
    f3 = np.zeros(s)
    f4 = np.zeros(s)

    j = 0
    for frame in FrameGenerator(Audio, frameSize = 1024, hopSize = 512):
        s1 = spectrum(w(frame))
        f1[j] = centroid(s1)
        f2[j], f3[j], f4[j] = distribution_shape(centrol_moment(s1))
        j = j+1
    
    return np.mean(f1[0:j]),np.var(f1[0:j]),np.mean(f2[0:j]),np.var(f2[0:j]),np.mean(f3[0:j]),np.var(f3[0:j]),np.mean(f4[0:j]),np.var(f4[0:j])
  
  
  
def Feature_Frequency_MFCC(Audio):  
    s = Audio.size/512+10
    f1 = np.zeros(s)
    f2 = np.zeros(s)

    mfccs1 = np.zeros([s,13])
    j = 0
    for frame in FrameGenerator(Audio, frameSize = 1024, hopSize = 512):
        if np.mean(frame)==0:
            continue
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        mfccs1[j,:] = mfcc_coeffs
        j = j+1
    
    mfccs_m = np.mean(mfccs1[0:j,:],axis=1)
    mfccs_v = np.var(mfccs1[0:j,:],axis=1)    
    
    
    return np.mean(mfccs_m[0:j]),np.var(mfccs_m[0:j]),np.mean(mfccs_v[0:j]),np.var(mfccs_v[0:j])
  
  
  

def Feature_Time(Audio):  
    s = Audio.size/512+10
    t1 = np.zeros(s)
    t2 = np.zeros(s)

    j = 0
    for frame in FrameGenerator(Audio, frameSize = 1024, hopSize = 512):
        t1[j] = np.mean(abs(frame))
        t2[j] = np.var(abs(frame))
        j = j+1
    
    return np.mean(t1[0:j]),np.var(t1[0:j]),np.mean(t2[0:j]),np.var(t2[0:j])
    
def Feature_Ratio(Audio, threshold):
    thr = pow(10,-threshold/20)
    s = Audio.size/32+2
    rms1 = np.zeros(s)
    j = 0
    for frame in FrameGenerator(Audio, frameSize = 64, hopSize = 32):
        rms1[j] = rms(frame)
        j = j+1

    rms_curve1 = low_pass(essentia.array(rms1))
    select_part = np.array([],dtype=int)
    for k in range(rms_curve1.size):
        if abs(rms_curve1[k])>thr:
            select_part = np.append(select_part,k)
 
    audio1 = rms_curve1[select_part]
    feature_base= np.mean(audio1)
    return feature_base
    
    
def attack_feature1(audio, threshold): 
    thr = pow(10,-threshold/20)
    signal = np.reshape(audio,[1,audio.size])
    signal = sumpf.Signal(signal,44100.0)
    
    obj = SignalEnvelope(signal)
    a = obj.GetEnvelope()
    b = np.array(a.GetChannels())
    b = np.reshape(b,b.size)
    

    y_top = peakutils.indexes(b, thres=thr, min_dist=100)
    y_bottom = peakutils.indexes(thr-b, thres=0, min_dist=100)
    
    top_label = np.array([y_top,np.zeros(y_top.size)])
    bottom_label = np.array([y_bottom,np.ones(y_bottom.size)])
    
    labels = np.concatenate((top_label,bottom_label), axis=1)
    sorted_labels = np.array(sorted(np.transpose(labels),key=itemgetter(0)),dtype='int')

    slop = []
    for i in range(1,int(sorted_labels.size/2)):
        j = i-1
        if sorted_labels[i,1]==1:
            j = i-1
        else:
            while(sorted_labels[j,0]==0):
                j-=1
            k = sorted_labels[i,0]-sorted_labels[j,0]
            slop.append((b[sorted_labels[i,0]]-b[sorted_labels[j,0]])/k)
        
    
    attack_feature_AVR = np.mean(slop)
    attack_feature_VAR = np.var(slop)
    
    return attack_feature_AVR, attack_feature_VAR
    
    
def release_feature1(audio, threshold):
    thr = pow(10,-threshold/20)
    signal = np.reshape(audio,[1,audio.size])
    signal = sumpf.Signal(signal,44100.0)
    
    obj = SignalEnvelope(signal)
    a = obj.GetEnvelope()
    b = np.array(a.GetChannels())
    b = np.reshape(b,b.size)

    y_top = peakutils.indexes(b, thres=thr, min_dist=100)
    y_bottom = peakutils.indexes(thr-b, thres=0, min_dist=100)
    
    top_label = np.array([y_top,np.zeros(y_top.size)])
    bottom_label = np.array([y_bottom,np.ones(y_bottom.size)])
    
    labels = np.concatenate((top_label,bottom_label), axis=1)
    sorted_labels = np.array(sorted(np.transpose(labels),key=itemgetter(0)),dtype='int')

    slop = []
    for i in range(sorted_labels.size/4,sorted_labels.size/2):
        j = i-1
        if sorted_labels[i,1]==0 and sorted_labels[j,1]==1:
            k = sorted_labels[i,0]-sorted_labels[j,0]
            slop.append((b[sorted_labels[i,0]]-b[sorted_labels[j,0]])/k)
        else:   
            i+=1
            j+=1
    
    release_feature_AVR = np.mean(slop)
    release_feature_VAR = np.var(slop)
    
    return release_feature_AVR, release_feature_VAR


def attack_feature2(Audio):
    s = Audio.size/128+5
    rms1 = np.zeros(s)
    j = 0
    for frame in FrameGenerator(Audio, frameSize = 256, hopSize = 128):
        rms1[j] = rms(frame)
        j = j+1

    rms_curve1 = low_pass(essentia.array(rms1))
    #n1 = peakutils.indexes(rms_curve1, thres=0.5, min_dist=70)
    #f3 = rms_curve1[n1[1]]
    
    x = np.linspace(0, rms_curve1.size-1, rms_curve1.size*16)
    x1 = np.linspace(0, rms_curve1.size-1, rms_curve1.size)
    f = interpolate.interp1d(x1,rms_curve1,kind="quadratic")
    y_rms = f(x)
    n2 = peakutils.indexes(y_rms, thres=0.4)[0]

    f3 = y_rms[n2]
    #n3 = np.count_nonzero(y_rms[0:np.argmax(y_rms)]<max(y_rms)*0.7)
    n3 = np.count_nonzero(y_rms[0:y_rms.size/10]<0.001)
    f1 = (n2-n3)

    f2 = np.mean(y_rms[n3:n2])
    #plt.plot(y_rms)

    return f1,f2,f3


    
def release_feature2(Audio):
    low_pass = LowPass(cutoffFrequency=500)
    s = Audio.size/128+5
    rms1 = np.zeros(s)
    j = 0
    for frame in FrameGenerator(Audio, frameSize = 256, hopSize = 128):
        rms1[j] = rms(frame)
        j = j+1

    rms_curve1 = low_pass(essentia.array(rms1))
    #n1 = peakutils.indexes(rms_curve1, thres=0.5, min_dist=70)
    #f3 = rms_curve1[n1[1]]
    #rms_curve1 = rms_curve[::-1]
    
    x = np.linspace(0, rms_curve1.size-1, rms_curve1.size*16)
    x1 = np.linspace(0, rms_curve1.size-1, rms_curve1.size)
    f = interpolate.interp1d(x1,rms_curve1,kind="quadratic")
    y_rms = f(x)
    #n2 = peakutils.indexes(y_rms, thres=0.4)[0]
    y_rms1 = max(y_rms)-y_rms[int(y_rms.size/2):y_rms.size]
    n2 = peakutils.indexes(y_rms1, thres=0.177)[0]
    
    f3 = y_rms1[n2]
    #n3 = np.count_nonzero(y_rms[0:np.argmax(y_rms)]<max(y_rms)*0.7)
    n3 = np.count_nonzero(y_rms1[int(y_rms1.size*0.9):y_rms.size]<0.001)
    f1 = (n2-n3)

    f2 = np.mean(y_rms1[0:n2])
    #plt.plot(y_rms)

    return f1,f2,f3
    
    
    
def attack_feature10(Audio1, Audio2):
    s = Audio1.size/128+5
    rms1 = np.zeros(s)
    j = 0
    for frame in FrameGenerator(Audio1, frameSize = 256, hopSize = 128):
        rms1[j] = rms(frame)
        j = j+1
    rms_curve1 = low_pass(essentia.array(rms1))
    
    s = Audio2.size/128+5
    rms2 = np.zeros(s)
    j = 0
    for frame in FrameGenerator(Audio2, frameSize = 256, hopSize = 128):
        rms2[j] = rms(frame)
        j = j+1
    rms_curve2 = low_pass(essentia.array(rms2))
    
    diff = rms_curve1/rms_curve2[0:rms_curve1.size]
    b = diff # for guitar
    #b = np.diff(diff) #for violin
    

    peak_point_1 = b[0:b.size/2].argmin()
    j = peak_point_1 - 10
    l = peak_point_1
    while(diff[l]<np.mean(diff[j:j+10])):
        l+=1

    f1 = (l-j)
    return f1
    
    
def release_feature10(Audio1, Audio2):
    s = Audio1.size/128+5
    rms1 = np.zeros(s)
    j = 0
    for frame in FrameGenerator(Audio1, frameSize = 256, hopSize = 128):
        rms1[j] = rms(frame)
        j = j+1
    rms_curve1 = low_pass(essentia.array(rms1))
    
    s = Audio2.size/128+5
    rms2 = np.zeros(s)
    j = 0
    for frame in FrameGenerator(Audio2, frameSize = 256, hopSize = 128):
        rms2[j] = rms(frame)
        j = j+1
    rms_curve2 = low_pass(essentia.array(rms2))
    
    diff = rms_curve1[::-1]/rms_curve2[rms_curve1.size:0:-1]
    b = np.diff(diff)
    
    #peaks = peakutils.indexes(-b, thres=0, min_dist=300)
    peak_point_1 = b[0:b.size/2].argmin()
    j = peak_point_1
    l = peak_point_1 + 10
    while(diff[l]<np.mean(diff[j-10:j+10])):
        l+=1

    f1 = (l-j)
    return f1
    
    
 
def Feature_RMS(Audio):  
    s = Audio.size/512+10
    t1 = np.zeros(s)

    j = 0
    for frame in FrameGenerator(Audio, frameSize = 1024, hopSize = 512):
        t1[j] = rms(frame)
        j = j+1
    
    return np.mean(t1[0:j]),np.var(t1[0:j])
  


'''
有问题！！！

'''

def Feature_Frequency_log(Audio):  
    s = Audio.size/512+10
    f1 = np.zeros(s)
    f2 = np.zeros(s)
    f3 = np.zeros(s)
    f4 = np.zeros(s)

    j = 0
    for frame in FrameGenerator(Audio, frameSize = 1024, hopSize = 512):
        if np.mean(frame)==0:
            continue
        s1 = spectrum(w(frame))
        f1[j] = centroid(np.log(s1))
        f2[j], f3[j], f4[j] = distribution_shape(centrol_moment(np.log(s1)))
        j = j+1

    return np.mean(f1[0:j]),np.var(f1[0:j]),np.mean(f2[0:j]),np.var(f2[0:j]),np.mean(f3[0:j]),np.var(f3[0:j]),np.mean(f4[0:j]),np.var(f4[0:j])
  

def attack_feature_loop(file_name):
    #file_name = '/Users/shengdi/Documents/code/database/loops/apple_loop/guitar/guitar12-Attack%d.wav' % (10*(i+2))
    loader = essentia.standard.MonoLoader(filename = file_name)
    Audio = loader()
    
    od_f = []
    for frame in FrameGenerator(Audio, frameSize = 1024, hopSize = 512):
        mag, phase, = c2p(fft(w(frame)))
        #pool.add('features.flux', od(mag, phase))
        od_f.append(od(mag, phase))

    onsets_flux = onsets(array([od_f]),  [ 1 ])
    
    s = Audio.size/128+5
    rms1 = np.zeros(s)
    j = 0
    for frame in FrameGenerator(Audio, frameSize = 256, hopSize = 128):
        rms1[j] = rms(frame)
        j = j+1

    rms_curve1 = low_pass(array(rms1))
    
    on_sets = np.array(onsets_flux*44100/128,dtype='int')
    
    if (on_sets[1]-on_sets[0])<25:
        first_note = rms_curve1[on_sets[1]:on_sets[2]]
    else:
        first_note = rms_curve1[on_sets[0]:on_sets[1]]
        
    x = np.linspace(0, first_note.size-1, first_note.size*16)
    x1 = np.linspace(0, first_note.size-1, first_note.size)
    f = interpolate.interp1d(x1,first_note,kind="quadratic")
    y_rms = f(x)
        
    start = np.argmin(y_rms[0:np.ceil(y_rms.size/5)])
    #stop = peakutils.indexes(y_rms[start:y_rms.size], thres=0.9, min_dist=50)[0]
    #stop = np.argmax(y_rms[start:y_rms.size])
    a = y_rms[start:y_rms.size]<max(y_rms[start:y_rms.size])*0.9
    stop = 0
    while(a[stop]!=0):
        stop = stop+1
    #print(stop)
    #plt.plot(y_rms)

    return stop, y_rms[stop], np.mean(y_rms[start:stop])



def attack_feature_loop_attacktime(file_name, on_sets):
    loader = essentia.standard.MonoLoader(filename = file_name)
    Audio = loader()
    s = Audio.size/128+5
    rms1 = np.zeros(s)
    j = 0
    for frame in FrameGenerator(Audio, frameSize = 256, hopSize = 128):
        rms1[j] = rms(frame)
        j = j+1
    rms_curve2 = rms1
    
    attack_time = np.zeros([on_sets.size-1,2])
    a = np.zeros([on_sets.size-1,3])

    for k in range(1,on_sets.size): 
        if (on_sets[k]-on_sets[k-1]<30):
            continue
        m = 0 if on_sets[k-1]-10<0 else on_sets[k-1]-10
        segment = rms_curve2[m:on_sets[k]]
        note = kz(segment, 5, 4)
        #start = np.argmin(note[0:note.size/7])
        peaks = np.size(peakutils.indexes(1-note[0:note.size/4]))
        start = peakutils.indexes(1-note[0:note.size/4])[0] if peaks!=0 else np.argmin(note[0:note.size/7])
        
        rms_curve1 = note[start:note.size]/max(note[start:note.size])
        n = point_finder(rms_curve1,0.9) if point_finder(rms_curve1,0.9)>3 else 3
        attack = rms_curve1[0:n]
        rest = rms_curve1[n:rms_curve1.size]
        predict = np.zeros(rms_curve1.size)
        popt1, pcov = curve_fit(func1, range(n), attack)
        predict[0:n] = func1(np.arange(n), *popt1)
        n_ = rms_curve1.size-np.argmax(rms_curve1) if (rms_curve1.size-np.argmax(rms_curve1))>3 else 3
        popt2, pcov = curve_fit(func2, range(n_), rms_curve1[rms_curve1.size-n_:rms_curve1.size])
        predict[rms_curve1.size-n_:rms_curve1.size] = func2(np.arange(n_), *popt2)
        attack_time[k-1,1] = np.mean(abs(predict - rms_curve1))
        p3 = np.count_nonzero(np.diff(rms_curve1[n:rms_curve1.size])>0)/float(rms_curve1.size-n)
        a[k-1,:] = popt1
        
        if (popt1[0]>0 and -popt1[1]/(2*popt1[0])>10) or (popt1[0]<0 and -popt1[1]/(2*popt1[0])<10) or (p3>0.7):
            continue
        
        x = np.linspace(0, note.size-1-start, (note.size-start)*16)
        x1 = np.linspace(0, note.size-start-1, note.size-start)
        f = interpolate.interp1d(x1,note[start:note.size],kind="quadratic")
        y_rms = f(x)

        attack_time[k-1,0] = point_finder(y_rms,0.9)-point_finder(y_rms,0.1)
        for q in range(attack_time.size/2):
            if (attack_time[q,0] < 3):
                attack_time[q,0] = 0

        b = attack_time[attack_time[:,0]!=0]
        order = np.argsort(b[:,1])
        ordered_pack = b[order]

        
    #f10[l,i] = sum(attack_time)/np.count_nonzero(attack_time)
    att = np.mean(ordered_pack[:,0]*(1/ordered_pack[:,1]))  #(1-np.arange(0,1,2.0/ordered_pack.size))) 

    return att


