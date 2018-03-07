# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:31:24 2017

@author: jkunert
"""
import numpy as np
from DMD import DMD
import matplotlib as mpl
import matplotlib.pyplot as plt
import os,time,sys
import h5py
import scipy as sci
import scipy.spatial.distance as ssd
import parseCIFTI as pC
from glob import glob
import h5py
import copy


def catImgs(f1,f2,fout,xshift=0):
    from PIL import Image
    im=map(Image.open,[f1,f2])
    w,h=zip(*(i.size for i in im))
    tw=sum(w)-xshift
    th=max(h)
    imout=Image.new('RGB',(tw,th))
    imout.paste(im[1],(im[0].size[0]-xshift,0))
    imout.paste(im[0],(0,0))
    imout.save(fout)

def bin2img(f,nzf,nb=40):
    A=np.zeros((nb*2,nb*2))
    A[nzf]=f
    return A

def sbinMode(phi,nb=40):
    #Function to bin cortex data spatially
    L,R,X,Lz,Rz=pC.grayord2surf(phi)
    Li=np.vstack([np.digitize(x,bins=np.linspace(np.min(x),np.max(x),nb),right=True) for x in L[:,:2].T]).T
    Ri=np.vstack([np.digitize(x,bins=np.linspace(np.min(x),np.max(x),nb),right=True) for x in R[:,:2].T]).T

    ###LEFT CORTEX    
    p=np.absolute(L[:,-1:])
    #Left exterior
    LF=np.zeros((nb,nb))
    LB=np.zeros((nb,nb))
    for n,ij in enumerate(Li):
        if Lz[n]<-29:
            if LF[ij[0],ij[1]]<p[n]:
                LF[ij[0],ij[1]]=p[n]
        else:
            if LB[ij[0],ij[1]]<p[n]:
                LB[ij[0],ij[1]]=p[n]
    LF=LF.T
    LF=np.fliplr(np.flipud(LF))
    LB=LB.T
    LB=np.flipud(LB)
    
    ###RIGHT CORTEX
    p=np.absolute(R[:,-1:])
    #Right exterior
    RF=np.zeros((nb,nb))
    RB=np.zeros((nb,nb))
    for n,ij in enumerate(Ri):
        if Rz[n]<45:
            if RF[ij[0],ij[1]]<p[n]:
                RF[ij[0],ij[1]]=p[n]
        else:
            if RB[ij[0],ij[1]]<p[n]:
                RB[ij[0],ij[1]]=p[n]
    RF=RF.T
    RF=np.flipud(RF)
    RB=RB.T
    RB=np.fliplr(np.flipud(RB))
    
    F=np.hstack([np.vstack([LF,LB]),np.vstack([RF,RB])])    
    return F
   
def binCortexData(x,nb=40):
    global R
    #load and sort data
    L,R,_,_,_=pC.parseCoords()
    nl=L.shape[0]
    nr=R.shape[0]
    xL=x[:nl,:]
    xR=x[nl:(nl+nr),:]
    Lz=L[:,0];L=L[:,1:]
    xL=xL[np.flipud(np.argsort(Lz)),:]
    L=L[np.flipud(np.argsort(Lz)),:]
    Rz=R[:,0];R=R[:,1:]
    xR=xR[np.argsort(Rz),:]
    R=R[np.argsort(Rz),:]
    Lz=np.flipud(np.sort(Lz))
    Rz=np.sort(Rz)
    
    x=np.vstack((xL,xR))    
    
    Li=np.vstack([np.digitize(bx,bins=np.linspace(np.min(bx),np.max(bx),nb),right=True) for bx in L[:,:2].T]).T
    Ri=np.vstack([np.digitize(bx,bins=np.linspace(np.min(bx),np.max(bx),nb),right=True) for bx in R[:,:2].T]).T
    
    LR=np.zeros((x.shape[0],1))
    LR[:nl]=1.0
    medial=np.hstack(((Lz>-29),(Rz<29)))
    
    #create index array giving [LR,medial?,x,y] bin coordinate of each voxel in x
    IJK=np.vstack((Li,Ri))
    IJK=np.hstack((LR,medial[:,None],IJK))
    IJK=IJK.dot(np.array([[nb**3.0],[nb**2.0],[nb],[1]]))
    R,J=np.unique(IJK,return_inverse=True)
    R=R.astype(int)
    #create sparse binning matrix of size (no. unique bins)x(original no. surface voxels)
    #multiplying with 'x' averages data within each bin
    import scipy.sparse as sparse
    binner=sparse.coo_matrix((np.ones((len(J),)),(J,np.arange(len(J)))),shape=(len(np.unique(J)),x.shape[0]))
    binner=sparse.diags(1.0/np.array(np.sum(binner,1)).ravel()).dot(binner)
    
    #average data across bins
    X=binner.dot(x)
    return X
    
    
def flat2mat(f):
    global R
    nb=int(np.max(R**(1/3.0)))
    Rit=R%nb
    y=copy.copy(Rit)
    x=((R-R%nb)/nb)%nb
    med=((R-R%nb**2)/nb**2)%nb
    LR=(R-R%nb**3)/nb**3
    
    F00=np.zeros((nb,nb))
    fx=f[(LR==1)&(med==0)]
    ix=y[(LR==1)&(med==0)]
    jx=x[(LR==1)&(med==0)]
    F00[ix,jx]=fx
    F00=np.fliplr(np.flipud(F00))
    
    F01=np.zeros((nb,nb))
    fx=f[(LR==0)&(med==0)]
    ix=y[(LR==0)&(med==0)]
    jx=x[(LR==0)&(med==0)]
    F01[ix%nb,jx%nb]=fx
    F01=np.flipud(F01)
    
    F10=np.zeros((nb,nb))
    fx=f[(LR==1)&(med==1)]
    ix=y[(LR==1)&(med==1)]
    jx=x[(LR==1)&(med==1)]
    F10[ix,jx]=fx
    F10=np.flipud(F10)
    
    F11=np.zeros((nb,nb))
    fx=f[(LR==0)&(med==1)]
    ix=y[(LR==0)&(med==1)]
    jx=x[(LR==0)&(med==1)]
    F11[ix%nb,jx%nb]=fx
    F11=np.fliplr(np.flipud(F11))  
    F=np.vstack((np.hstack((F00,F01)),np.hstack((F10,F11))))   
    return F
    

def mat2flat(F):
    global R
    nb=int(np.max(R**(1/3.0)))
    Rit=R%nb
    y=copy.copy(Rit)
    x=((R-R%nb)/nb)%nb
    med=((R-R%nb**2)/nb**2)%nb
    LR=(R-R%nb**3)/nb**3
    
    f=[]
    
    w=F.shape[0]/2
    Fs=[np.flipud(np.fliplr(F[:w,:w])),np.flipud(F[:w,w:]),np.flipud(F[w:,:w]),np.flipud(np.fliplr(F[w:,w:]))]
    n=0
    for medk in [0,1]:
        for lrk in [1,0]:
            Fij=Fs[n]
            n+=1
            ix=y[(LR==lrk)&(med==medk)]
            jx=x[(LR==lrk)&(med==medk)]
            fx=Fij[ix,jx]
            f.append(fx)
 
    f=np.concatenate((f[1],f[3],f[0],f[2]))
    return f
    
    
def vizSmoothMode(p):
    P=flat2mat(p)
    plt.figure(facecolor=(1,1,1))
    plt.get_current_fig_manager().window.setGeometry(775,83,1130,932)
    P[P==0]=np.nan
    plt.imshow(np.abs(P),interpolation='None',cmap='coolwarm')
    plt.colorbar()
def loadModes(fdir='./results/binnedModes/'):
    with h5py.File(fdir+'DMDmodes.h5','r') as hf:
        Phi=np.array(hf['Phi'])
        freq=np.array(hf['freq'])
        power=np.array(hf['power'])
        ux=np.array(hf['ux'])
    with h5py.File(fdir+'BinnedModes.h5','r') as hf:
        F=np.array(hf['F'])
    return Phi,freq,power,ux,F
    
def loadSpatialClusters():
    with h5py.File(fdir+'HClusters.h5','r') as hf:
        C=np.array(hf['C'])
        ix=np.array(hf['ix'])
    return C,ix
  
def initR():
    binCortexData(np.ones((91282,1)))
    return None