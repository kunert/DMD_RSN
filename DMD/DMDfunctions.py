# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:31:55 2017

@author: jkunert
"""

import numpy as np
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py

def DMD(C,n=[],p=[],h=[]):
    #inputs data array C -- rows are variables, columns are timepoints
    #optional inputs:
    #n = number of DMD modes to keep; if n=[] (default) then uses number of variables in C
    #p = if set and n==0, then uses first n modes to capture fraction 'p' of energy
    #h = timestep between measurements, for calculating frequencies in Hz
    #     (if h is unset, uses default of h=(14*60+33)/1200. for HCP data)
    
    #split data matrix into two subarrays shifted by one timepoint
    #such that X[:,i+1]=Xp[:,i]
    X=copy.copy(C)
    X+=-np.mean(X,1)[:,None]
    Xp=X[:,1:]
    X=X[:,:-1]

    #compute SVD of data matrix
    U,S,V=np.linalg.svd(X,full_matrices=False)
    

    if (n==[])&(p!=[]):
        n=np.where((np.cumsum(S)/np.sum(S))>=p)[0][0]+1
        print 'KEEPING {:} MODES TO CAPTURE {:} OF ENERGY'.format(n,p)
    if n==[]:
        n=X.shape[0]
        
    Ut=U[:,:n]
    Sinv=np.diag(1./S[:n])
    Vt=V[:n].T
    
    #compute reduced-dimensional representation of A-matrix
    Ap=(Ut.T).dot(Xp.dot(Vt.dot(Sinv)))
    #weight A by singular values
    Ah=np.diag(S[:n]**-0.5).dot(Ap.dot(np.diag(S[:n]**0.5)))
    #compute eigendecomposition of weighted A-matrix
    w,v=np.linalg.eig(Ah)
    v=np.diag(S[:n]**0.5).dot(v)
    
    #compute DMD modes from eigenvectors
    Phi=Xp.dot(Vt.dot(Sinv.dot(v)))
    #computed this way, DMD modes are not normalized; norm gives power of mode in data
    power=np.real(np.sum(Phi*Phi.conj(),0))
     
    #HCP900 Data reference manual gives 1200 frames per run, 14:33 run duration.. gives approximate
    # seconds/frame as...
    if h==[]:
        h=(14*60+33)/1200.
    #use h to convert complex eigenvalues into corresponding oscillation frequencies
    freq=np.angle(w)/(2*np.pi*h)
    
    return Phi,power,freq
    
    
if __name__=='__main__':
  with h5py.File('test_data.h5','r') as hf:
    Xr=np.array(hf['Xr'])
    P=np.array(hf['P'])
    p=np.array(hf['p'])
    f=np.array(hf['f'])
  nmodes=3
  for k,frame0 in enumerate([0,10]):
    X=Xr[:,frame0:(frame0+50)]
    phik,pt,ft=DMD(X,n=nmodes)
    #check that we get same results as previous version of code
    assert np.all(phik==P[k])
    assert np.all(pt==p[k])
    assert np.all(ft==f[k])
  print 'Consistency test passed!'

      
    