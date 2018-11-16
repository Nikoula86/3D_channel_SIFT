#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:37:28 2018

@author: ngritti
"""

import sys
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imsave

def load_images(fileTemplate,shape):
    '''Load raw data as nD numpy array.
    
    Args:
        fileTemplate (str): absolute or relative path to rawData.
            The srt should contain one '*' character to identify all the files.
        shape (tuple): expected shape of a single stack.
        
    Returns:
        imgs (nD array): numpy array containing all stacks. By default outputs
            in uint16 type. Dimension is input shape + 1.
        axID (str): axed identification character.
    
    '''
    fList=glob(fileTemplate)
    fList.sort()
    print('Files detected: ',len(fList))
    if '.raw' in fileTemplate:
        print('Files format: .raw, loading data...')
        imgs = np.stack( [ np.reshape(np.frombuffer(open(i,'rb').read(),np.uint16)[:np.prod(shape)],shape) for i in tqdm(fList) ] )
    elif '.tif'in fileTemplate:
        print('Files format: .tif, loading data...')
        imgs = np.stack( [ imread(i) for i in tqdm(fList) ] )
    if len(imgs.shape)==4:
        axID = 'CZYX'
    elif len(imgs.shape)==3:
        axID = 'CXY'
    print('axID: ',axID)
    check_mem(imgs)
    return imgs, axID

def check_mem(array):
    '''Check array memory.
    
    This simply outputs shape, type and memory usage of the input numpy array.
    
    Args:
        array: input numpy array to be checked.
        
    Returns:
        
    '''
    print('Data shape: ', array.shape)
    print('Data type: ', array.dtype)
    print('Data memory (MB): ', sys.getsizeof(array)/(1024**2))    
    print('\n')
    

def get_mip(stack,axID,s,checkMem=True):
    '''Maximum intensioty projection.
    
    Args:
        stack (ndarray): input nd numpy array.
        axID ('str'): axis identifiers of the input array.
        s ('str'): single character identifying the axis over which to compute
            the maximum projection.
        checkMem (bool, optional): check memory of output array. Default:True.
            
    Returns:
        outstack ((n-1)d array): MIP of the input array over the s axis.
        
    '''
    print('Computing maximum projection along %s axis...'%s)
    if s not in axID:
        raise ValueError('Please provide a valid mip ax! s not in axID!!!')
    if len(s)!=1:
        raise ValueError('Please provide a valid mip ax! len(2)!=1!!!')
    i = axID.index(s)
    outstack = np.max(stack,i)
    if checkMem:
        check_mem(outstack)
    return outstack

def renormalize(instack,perc=99.7,checkMem=True,visual=False):
    '''Renormalize input to percentiles and scale to 8bit.
    
    Args:
        instack (nd array): input array. Can be 3d or 4d.
        perc (float, optional): max percentile. Default: 99.7.
        checkMem (bool, optional): check memory of output array. Default:True.
        visual (bool, optional): flag to visualize all channels (or the mip 
               of every channels).
        
    Returns:
        outstack (ndarray): 8bit version of input stack.
        
    '''
    print('Renormalizing to 8bit...')
    outstack = np.zeros(instack.shape).astype(np.float64)
    for i in tqdm(list(range(instack.shape[0]))):
        _min = np.min(instack[i])
        _max = np.percentile(instack[i],perc)
        outstack[i] = np.clip(instack[i],_min,_max)
        outstack[i] = (2**8-1) * ((outstack[i]-_min)/(_max-_min))
    outstack = outstack.astype(np.uint8)
    if checkMem:
        check_mem(outstack)
    if visual:
        fig,ax = plt.subplots(figsize=(outstack.shape[0]*6,6),nrows=1,ncols=outstack.shape[0])
        fig.suptitle('Renormalized MIPs')
        ax = ax.flatten()
        ch = ['ch%02d'%i for i in range(instack.shape[0])]
        for i in range(outstack.shape[0]):
            plotImg = outstack[i]
            if len(outstack[i].shape)==3:
                plotImg = get_mip(outstack[i],'ZYX','Z',checkMem=False)
            ax[i].imshow(plotImg,cmap='gray',vmin=0,vmax=255)
            ax[i].set_xlabel(ch[i])
        plt.show()
    return outstack

def resize_array(instack,upsampling):
    '''Resize array over the first dimension.
    
    Args:
        instack (ndarray): input nd array.
        upsampling (int): rescaling dimension factor.
    
    Returns:
        outstack (nd array): resized array. Same dtype as input array.
        
    '''
    _type=instack.dtype
    from skimage.transform import resize
    size=list(instack.shape)
    size[0] *= int(upsampling)
    outstack = resize(instack.astype(np.float64),output_shape=size).astype(_type)
    return outstack
        
#%%
'''
source:
https://ianlondon.github.io/blog/how-to-sift-opencv/
'''
import cv2
def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp, ax):
    return ax.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy(),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))


#%%

def compute_features(instack,visual=False):
    '''Compute SIFT features.
    
    Loops over the first dimension of the input stack and computes all the SIFT
    features. Because SIFT only works on 2D images, the input stack MUST be 3D.
    
    Args:
        instack (3D array): input images.
        visual (bool, optional): flag to visualize features. Default: False.
        
    Returns:
        kp (list): list of keypoints found in every channels.
            NOTE: len(kp) = instack.shape[0].
        desc (list): list of descriptors for every keypoint.
            NOTE: len(desc) = instack.shape[0].
        
    '''
    print('Computing features...')
    if len(instack.shape)!=3:
        raise ValueError('Can\'t run SIFT over 3D images! instack must be 3D!')
    kp = [[] for i in instack]
    desc = [[] for i in instack]
    for i in range(instack.shape[0]):
        kp[i], desc[i] = gen_sift_features(instack[i])
        
    if visual:
        fig,ax = plt.subplots(figsize=(instack.shape[0]*6,6),nrows=1,ncols=instack.shape[0])
        fig.suptitle('Detected features in every MIPS')
        ax = ax.flatten()
        ch = ['ch%02d'%i for i in range(instack.shape[0])]
        for i in range(instack.shape[0]):
            ax[i].set_xlabel(ch[i])
            show_sift_features(instack[i], instack[i], kp[i], ax[i]);
        plt.show()
    return (kp,desc)

def match_features(instack,kps,descs,N=25,visual=False):
    '''Matching features.
    
    This function matches the features in the first channel to features in 
    every other channel.
    
    Args:
        instack (3D array): input array.
        kps (list): keypoints.
        descs (list): descriptors.
        N (int, optional): number of best features to keep. Default: 25.
        visual (bool, optional): visualize feature matching. default: False
        
    Returns:
        matches (list): features matched.
        
    '''
    print('Matching features...')
    bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
    matches = [[] for i in instack]
    for i in range(instack.shape[0]):
        m = bf.match(descs[0],descs[i])
        m = sorted(m,key=lambda x: x.distance)
        matches[i] = m[:N]
    
    if visual:
        fig,ax = plt.subplots(figsize=(6,instack.shape[0]*3),nrows=instack.shape[0],ncols=1)
        fig.suptitle('Matching features')
        ax = ax.flatten()
        ch = ['ch00-ch%02d'%i for i in range(instack.shape[0])]
        for i in range(instack.shape[0]):
            img = cv2.drawMatches(instack[0],kps[0],instack[i],kps[i],matches[i],instack[i].copy(),flags=2)
            ax[i].imshow(img)
            ax[i].set_ylabel(ch[i])
        plt.show()
    return matches

def compute_homography(matches,kps):
    '''Compute the image transformation to register the corresponding images.
    
    This function always uses the first element of the list as reference.
    I.e. it compute the homography to align channel 1,2... to channel 0.
    
    Args:
        matches (list): matching features.
        kps (list): keypoints in the images.
        
    Returns:
        h (list): every element contains the 3D array that defines the transformation.
                  NOTE: h[0] is always the identity matrix.
    
    '''
    print('Computing Homography...')
    h = [[]for i in kps]
    mask = [[]for i in kps]
    for i in range(len(kps)):
        m = matches[i]
        p1 = np.zeros((len(m),2),dtype=np.float32)
        p2 = np.zeros((len(m),2),dtype=np.float32)
#        print(p1.shape,p2.shape)
        for j, match in enumerate(m):
    #        print(j)
            p1[j,:] = kps[0][match.queryIdx].pt
            p2[j,:] = kps[i][match.trainIdx].pt
        h[i],mask[i] = cv2.findHomography(p2,p1,cv2.RANSAC)
    return h

#%%
    
def save_homography(h,name,path):
    basedir = os.path.join(*path.split('/')[:-1])+'/Registered'
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    np.savez(basedir+'/'+fileStruct.split('/')[-1].split('.')[0]+'_'+name,h=h)

def save_registered_stacks(stacks,name,path):
    basedir = os.path.join(*path.split('/')[:-1])+'/Registered'
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    for i in tqdm(range(stacks.shape[0])):
        imsave(basedir+'/'+name%i,stacks[i])
    
def XY_register_stacks(stacks,h,fileStruct,save=False,visual=False):
    ''' XY registration transformation.
    
    Args:
        stacks (nd array): instup nd array. Can be 3D (mip, axID='CXY') 
                           or 4D (full stack, axID='CZYX'). If 3D, computes the transformation
                           on every channel. If 4D, computes the transformation on single XY planes.
        h (list): homography. List of 3x3 numpy arrays.
        fileStruct (str): input fileStructure. Used in case save=True.
        save (bool, optional): saving images in subfolder. Default: False.
        visual (bool, optional): visualize output. Default: False.
    
    Returns:
        imgsXYreg (ndarray): registered images.
    '''
    
    print('Registering stacks in XY...')
    imgsXYreg = 0*stacks
    if len(stacks.shape)==4:
        height,width = stacks[0,0].shape
        for i in tqdm( range(stacks.shape[1]) ):
            for j in range(len(h)):
                imgsXYreg[j,i] = cv2.warpPerspective(stacks[j,i],h[j],(width,height))
    elif len(stacks.shape)==3:
        height,width = stacks[0].shape
        for j in range(len(h)):
            imgsXYreg[j] = cv2.warpPerspective(stacks[j],h[j],(width,height))
    if save:
        print('\nSaving XY-registered stacks...')
        name = (fileStruct.split('/')[-1].split('.')[0]+'_XY.tif')
        name = name.replace('*','%02d')
        print(name)
        save_registered_stacks(imgsXYreg,name,fileStruct)
    if visual:
        fig,ax = plt.subplots(figsize=(imgsXYreg.shape[0]*6,6),nrows=1,ncols=imgsXYreg.shape[0])
        ax = ax.flatten()
        ch = ['ch%02d'%i for i in range(stacks.shape[0])]
        if len(imgsXYreg.shape)==4:
            fig.suptitle('XY registered images - single plane in the middle of the stack')
            plotImg = imgsXYreg[:,int(imgsXYreg.shape[1]/2),...]
        if len(imgsXYreg.shape)==3:
            fig.suptitle('XY registered MIP')
            plotImg = imgsXYreg
        for i in range(plotImg.shape[0]):
            ax[i].imshow(plotImg[i],cmap='gray')
            ax[i].set_xlabel(ch[i])
        plt.show()
    return imgsXYreg
        
def YZ_register_stacks(stacks,h,fileStruct,upsampling=4,save=True, visual=False):
    ''' YZ registration transformation.
    
    Args:
        stacks (nd array): instup nd array. Can be 3D (mip, axID='CYZ') 
                           or 4D (full stack, axID='CZYX'). If 3D, computes the transformation
                           on every channel. If 4D, computes the transformation on single YZ planes.
        h (list): homography. List of 3x3 numpy arrays.
        fileStruct (str): input fileStructure. Used in case save=True.
        upsampling (int): Z upsampling factor.
        save (bool, optional): saving images in subfolder. Default: False.
        visual (bool, optional): visualize output. Default: False.
    
    Returns:
        imgsYZreg (ndarray): registered images.
    '''
    print('Registering stacks in YZ...')
    imgsReg = 0*stacks
    if len(stacks.shape)==4:
        height,width = stacks[0,...,0].shape
        for i in tqdm( range(stacks.shape[-1]) ):
            for j in range(len(h)):
                tmp = resize_array(stacks[j,...,i],upsampling=upsampling)
                imgsReg[j,...,i] = cv2.warpPerspective(tmp,h[j],(width,upsampling*height))[::upsampling]
    elif len(stacks.shape)==3:
        height,width = stacks[0,...].shape        
        for j in range(len(h)):
            tmp = resize_array(stacks[j,...],upsampling=upsampling)
            imgsReg[j,...] = cv2.warpPerspective(tmp,h[j],(width,upsampling*height))[::upsampling]
    if save:
        print('\nSaving YZ-registered stacks...')
        name = (fileStruct.split('/')[-1].split('.')[0]+'_YZ.tif')
        name = name.replace('*','%02d')
        print(name)
        save_registered_stacks(imgsReg,name,fileStruct)
    if visual:
        fig,ax = plt.subplots(figsize=(6,imgsReg.shape[1]*imgsReg.shape[0]*6*2/imgsReg.shape[2]),
                              nrows=imgsReg.shape[0],ncols=1)
        ax = ax.flatten()
        ch = ['ch%02d'%i for i in range(imgsReg.shape[0])]
        if len(imgsReg.shape)==4:
            fig.suptitle('YZ registered images - single plane in the middle of the stack')
            plotImg = imgsReg[:,...,int(imgsReg.shape[2]/2)]
        if len(imgsReg.shape)==3:
            fig.suptitle('YZ registered MIP')
            plotImg = imgsReg
        for i in range(plotImg.shape[0]):
            ax[i].imshow(plotImg[i],cmap='gray')
            ax[i].set_xlabel(ch[i])
        plt.show()
    return imgsReg
  
#%%
def compute_MIP_XY(stacks, axID, fileStruct,
                   save=False, visual=False, checkMem=True):
    print('\n\tXY REGISTRATION.\n')
    mips = get_mip(stacks,axID,'Z',checkMem=checkMem)
    mips = renormalize(mips,perc=99.7,checkMem=checkMem,visual=visual)
    kps,descs = compute_features(mips,visual=visual)
    matches = match_features(mips,kps,descs,visual=visual)
    h = compute_homography(matches,kps)
    del mips,kps,descs,matches
    save_homography(h,'XY_homography.npz',fileStruct)
    regStack = XY_register_stacks(stacks,h,fileStruct,save=save)
    return regStack

def compute_MIP_YZ(stacks, axID, fileStruct,
                   save=True, visual=False, checkMem=True, upsampling=4):
    print('\n\tYZ REGISTRATION.\n')
    mips = get_mip(stacks,axID,'X',checkMem=checkMem)
    mips = np.stack([resize_array(m,upsampling) for m in mips])
    mips = renormalize(mips,perc=99.7,checkMem=checkMem,visual=visual)
    kps,descs = compute_features(mips,visual=visual)
    matches = match_features(mips,kps,descs,visual=visual)
    h = compute_homography(matches,kps)
    del mips,kps,descs,matches
    save_homography(h,'YZ_homography.npz',fileStruct)
    regStack = YZ_register_stacks(stacks,h,fileStruct,upsampling=upsampling,save=save)
    return regStack

def register_data(fileStruct, shape,
                  saveXY=False, visualXY=False, checkMemXY=True,
                  saveYZ=True, visualYZ=False, checkMemYZ=True, Zupsampling=4):
    rawData,axID = load_images(fileStruct,shape)
    regData = compute_MIP_XY(rawData,axID,fileStruct,saveXY,visualXY,checkMemXY)
    regData = compute_MIP_YZ(regData,axID,fileStruct,saveYZ,visualYZ,checkMemYZ,upsampling=Zupsampling)
    return (rawData, regData)

#%%
    
if __name__=='__main__':
    fileStructs = [ 'testSample/ch*_ill00.tif' ]
    
    for fileStruct in fileStructs:
        print('\nSOURCE FOLDER:\n'+fileStruct+'\n')
        raw,reg=register_data( fileStruct, shape=(200,2048,2048),
                              checkMemXY=False, checkMemYZ=False, 
                              saveXY=False, saveYZ=True,
                              visualXY=False, visualYZ=False,
                              Zupsampling=4 )



        