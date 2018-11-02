

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 13:48:52 2018

@author: danlinpeng
"""
import mne
import numpy as np
import torch
from scipy import signal, fftpack
import matplotlib.pyplot as plt

from mne.time_frequency.tfr import morlet
from mne.viz import plot_filter, plot_ideal_filter

import mne
import pickle

    
def load_data(batch_size,load_type='train'):
    features = np.load('image_feature.npy')
    event = [10,15,30,35,39]
    im_num = [n for _ in range (5) for n in range(1,51)]
    for subject_num in [1,2,3,5,6,7,8,9,10,11]:
            #get data
            filename = 'S'+str(subject_num)+'Exp1.pth'
            data = torch.load(filename)
            #pick event
            labels = data['y']
            data = data['x']
           
            index = np.isin(labels,event)
            index = np.where(index)
            data = data[index]
            labels = np.array(labels)
            labels = labels[index]
            #get image feature
            img = features[:data.shape[0],0,:] 
            im_num = im_num[:data.shape[0]] 
            #cut the data
            data = data[:,:,150:350] 
            
            #normalise
            for i in range(64):
                num = (data.shape[0]*data.shape[2])
                mean = data[:,i,:].sum()/num
                data[:,i,:]-=mean
                #remove std
                std = np.sqrt((data[:,i,:]**2).sum()/num)
                data[:,i,:]/=std
            if subject_num ==1:
                train_data=data #data
                im=img #imge feature
                l=labels #image label
                im_n=im_num #image number
            else:
                train_data=np.vstack((train_data,data))
                im=np.concatenate([im,img])
                l=np.concatenate([l,labels])
                im_n=np.concatenate([im_n,im_num])


    index = dict()
    #split dataset
    im_n = np.array(im_n)
    index['test'] = np.where(im_n >45 )
    index['valid'] = np.where((40<im_n) & (im_n <=45))
    index['train']= np.where(im_n<=40)
    train_data = np.transpose(train_data,(0,2,1))

    data = dict()
    for split in ('train','valid','test'):
        data[split]=[]
        split_data = train_data[index[split],:,:][0]
        split_target_feature = im[index[split],:][0]
        split_label = l[index[split]]
        split_image_number = im_n[index[split]]
        if load_type == 'train':
        #permutation
            permutation = np.random.permutation(split_data.shape[0])
            split_data = split_data[permutation,:,:] #data
            split_target_feature = split_target_feature[permutation] 
            split_label = split_label[permutation] #class label
            split_image_number = split_image_number[permutation] #image number
        
        for e in range(max(int(split_data.shape[0]/batch_size),1)):
            d = dict()
            d['data']=split_data[e*batch_size:(e+1)*batch_size]
            d['target_feature']=split_target_feature[e*batch_size:(e+1)*batch_size]
            
            d['label']=split_label[e*batch_size:(e+1)*batch_size]
            d['img_num']=split_image_number[e*batch_size:(e+1)*batch_size]
            data[split].append(d)
            
         
    return data        
        
a = load_data(32)
