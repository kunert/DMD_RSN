# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:31:24 2017

@author: jkunert
"""
import numpy as np
import h5py
import nibabel as nib

def loadGrayord(fname,norm=True):
    img=nib.load(fname);data=img.get_data()
    data=data[0,0,0,0,:,:].T
    if norm:
        data=(data-np.mean(data,1)[:,None])/np.std(data,1)[:,None]
    return data

def generateBinIndices(nb=40):
    #load and sort data
    with h5py.File('./grayordCoords.h5','r') as hf:
      L=np.array(hf['L'])
      R=np.array(hf['R'])
    #split z-coordinates of grayords from x,y coordinates
    Lz=L[:,0];L=L[:,1:]
    Rz=R[:,0];R=R[:,1:]
    #for both x,y coordinates on both left/right side, use np.digitize to assign coordinates into bins
    #where bin boundaries are equally spaced between minimum and maximum values
    Li=np.vstack([np.digitize(bx,bins=np.linspace(np.min(bx),np.max(bx),nb),right=True) for bx in L[:,:2].T]).T
    Ri=np.vstack([np.digitize(bx,bins=np.linspace(np.min(bx),np.max(bx),nb),right=True) for bx in R[:,:2].T]).T
    
    #create index array giving [L/R?,medial?,x_bin,y_bin] for each grayordinate
    #the last two columns (the bin coordinates) come from stacking Li,Ri on top of each other
    #so the "L/R"? label (0=Left,1=Right) 
    LR=np.zeros((L.shape[0]+R.shape[0],1))
    LR[:L.shape[0]]=1.0
    #by eye, the division between medial/lateral grayordinates is at about z=+/-29
    medial=np.hstack(((Lz>-29),(Rz<29)))
    
    #combine into index array = [L/R?,medial?,x_bin,y_bin]
    IJK=np.vstack((Li,Ri))
    IJK=np.hstack((LR,medial[:,None],IJK))
    #generate a single index which encodes all four columns
    IJK=IJK.dot(np.array([[nb**3.0],[nb**2.0],[nb],[1]]))
    #find unique values of the bin index, each of which uniquely corresponds to a bin with a
    #particular L/R, medial/lateral label and (x,y) bin coordinate
    #'binIndices' gives the bin index for each bin
    #'J' gives the bin number for each grayordinate
    binIndices,J=np.unique(IJK,return_inverse=True)
    binIndices=binIndices.astype(int)
    return binIndices,J

def binCortexData(x,nb=40):
    import scipy.sparse as sparse
    #create sparse binning matrix of size (no. unique bins)x(original no. surface voxels)
    binIndices,J=generateBinIndices(nb=nb)
    binner=sparse.coo_matrix((np.ones((len(J),)),(J,np.arange(len(J)))),shape=(len(np.unique(J)),x.shape[0]))
    binner=sparse.diags(1.0/np.array(np.sum(binner,1)).ravel()).dot(binner)
    #multiplying with 'x' averages data within each bin
    X=binner.dot(x)
    return X
    
    
def flat2mat(f,nb=40):
    f=f.ravel()
    #get bin indices
    binIndices,_=generateBinIndices(nb=nb)
    ##test that nb is correct size
    if len(binIndices)!=len(f):
      raise Exception('Input vector length does not match number of bins for nb={:}. Check that argument "nb" for flat2mat matches "nb" used for initial binning.'.format(nb))      
    #bin indices encode L/R label, medial label, and x/y coordinates of bin
    #so extract these values:
    y=binIndices%nb
    x=((binIndices-binIndices%nb)/nb)%nb
    med=((binIndices-binIndices%nb**2)/nb**2)%nb
    LR=(binIndices-binIndices%nb**3)/nb**3
    
    #for different combinations of L/R, medial/lateral,
    #assign bin values to bin location in subarray
    #flipping as necessary for correct visualization
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
    
    #stack subarrays into single array and return
    F=np.vstack((np.hstack((F00,F01)),np.hstack((F10,F11))))
    
    return F
    

def mat2flat(F):
    #infer nb from matrix shape
    nb=F.shape[0]/2
    #bin indices encode L/R label, medial label, and x/y coordinates of bin
    #so again extract these values:
    binIndices,_=generateBinIndices(nb=nb)
    nb=int(np.max(binIndices**(1/3.0)))
    y=binIndices%nb
    x=((binIndices-binIndices%nb)/nb)%nb
    med=((binIndices-binIndices%nb**2)/nb**2)%nb
    LR=(binIndices-binIndices%nb**3)/nb**3
    
    f=[]
    
    #split apart and un-flip subarrays corresponding to different L/R, med/lat labels
    Fs=[np.flipud(np.fliplr(F[:nb,:nb])),np.flipud(F[:nb,nb:]),np.flipud(F[nb:,:nb]),np.flipud(np.fliplr(F[nb:,nb:]))]
    n=0
    for medk in [0,1]:
        for lrk in [1,0]:
            Fij=Fs[n]
            n+=1
            #get x,y coordinates of bins with matching labels
            ix=y[(LR==lrk)&(med==medk)]
            jx=x[(LR==lrk)&(med==medk)]
            #extract bins at (ix,jx) coordinates into flattened array
            fx=Fij[ix,jx]
            f.append(fx)
    #combine flattened subarrays in the right order
    f=np.concatenate((f[1],f[3],f[0],f[2]))
    return f

  
if __name__=='__main__':
  x=np.random.randn(59412,5)
  nb=40
  X=binCortexData(x)
  F=flat2mat(X[:,0])
  assert np.all(mat2flat(F)==X[:,0])
  print "Utility functions: smoke test passed!"