# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:31:55 2017

@author: jkunert
"""

import numpy as np
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt


def combineDataList(Xl):
    X=[]
    Xp=[]
    if type(Xl)==np.ndarray:
        Xl=[Xl]
    for Xk in Xl:
        Xk+=-np.mean(Xk,1)[:,None]
        Xpk=Xk[:,1:]
        Xk=Xk[:,:-1]
        X.append(Xk)
        Xp.append(Xpk)
    X=np.hstack(X)
    Xp=np.hstack(Xp) 
    return X,Xp

def shiftstack(Xl):
    SS=[]
    if type(Xl)==np.ndarray:
        Xl=[Xl]
    for Xk in Xl:
        SS.append(np.concatenate((Xk[:,:-1],Xk[:,1:]),0))
    return SS

def DMD(C,n=[],ploton=True,p=[],h=[]):
    #inputs data matrix C -- rows are variables, columns are timepoints
    # C can either be a numpy array or a list of numpy arrays
    #optional inputs:
    #n = number of DMD modes to keep; if n=[] (default) then uses number of variables in C
    Xl=copy.copy(C)
    X,Xp=combineDataList(Xl)    
    
    U,S,V=np.linalg.svd(X,full_matrices=False)
    
    #print np.cumsum(S)/np.sum(S)
    
    if (n==[])&(p!=[]):
        n=np.where((np.cumsum(S)/np.sum(S))>=p)[0][0]+1
        print 'KEEPING {:} MODES TO CAPTURE {:} OF ENERGY'.format(n,p)
    if n==[]:
        n=X.shape[0]
    Ut=U[:,:n]
    Sinv=np.diag(1./S[:n])
    Vt=V[:n].T
    
    Ap=(Ut.T).dot(Xp.dot(Vt.dot(Sinv)))
    
    Ah=np.diag(S[:n]**-0.5).dot(Ap.dot(np.diag(S[:n]**0.5)))
    #Ap=Ap[:n,:][:,:n]
    #Ah=Ap
    w,v=np.linalg.eig(Ah)
    v=np.diag(S[:n]**0.5).dot(v)
    
    Phi=Xp.dot(Vt.dot(Sinv.dot(v)))
    
    #HCP900 Data reference manual gives 1200 frames per run, 14:33 run duration.. gives approximate
    # seconds/frame as...
    if h==[]:
        h=(14*60+33)/1200.
    
    power=np.real(np.sum(Phi*Phi.conj(),0))
    
    freq=np.angle(w)/(2*np.pi*h)
    
    if ploton is True:
        plt.figure(facecolor=(1,1,1))
        plt.get_current_fig_manager().window.setGeometry(965,251,787,560)
        plt.plot(freq[freq>=0],power[freq>=0],'.')
        for k,f in enumerate(freq[freq>=0]):
            plt.plot([f,f],[0,power[freq>=0][k]],c=(0,0,0))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        #plt.savefig('./atlasDMD'+str(n)+'.png')
    return Phi,w,v,power,freq,Ap
    
    
if __name__=='__main__':
    pass
    #C=parcellateData('./100408/tfMRI_RELATIONAL_LR.nii.gz')
    #Phi,w,v,power,freq=DMD(shiftstack(shiftstack(C)),n=246)
    """
    phik,w,v,pt,ft=DMD(Xr[:,:20],p=1.0,ploton=False)
    b=np.linalg.pinv(phik).dot(Xr[:,0][:,None])

    tn=2

    x0=np.real(phik.dot(b*np.exp(w*tn)[:,None]))
    plt.cla()
    plt.plot(x0,Xr[:,tn],'.')
    """