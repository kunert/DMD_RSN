# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:51:47 2017

@author: jkunert
"""
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os,sys
import h5py

def parseCoords():
    fpath=os.path.join('/'.join(os.path.abspath(__file__).split('/')[:-1]),'coordInfo.h5')
    with h5py.File(fpath,'r') as hf:
      dL=np.array(hf['dL'])
      dR=np.array(hf['dR'])
      idx=np.array(hf['idx'])
      SCbounds=np.array(hf['SCbounds'])
      SClabels=np.array(hf['SClabels'])
    return dL,dR,idx,SCbounds,SClabels


def grayord2surf(x):
    dL,dR,idx,_,_=parseCoords()
    nl=dL.shape[0]
    nr=dR.shape[0]
    nx=idx.shape[0]
    if len(x)!=(nl+nr+nx):
        raise Exception("Mismatch between length of input vector and number of coordinates: expected {:} grayordinates".format(nl+nr+nx))
    xL=x[:nl]
    xR=x[nl:(nl+nr)]
    xX=x[(nl+nr):]
    L=np.concatenate((dL[:,1:],xL[:,None]),1)
    R=np.concatenate((dR[:,1:],xR[:,None]),1)
    X=np.concatenate((idx[:,:3],xX[:,None]),1)

    #sort by depth
    Lz=dL[:,0];
    L=L[np.flipud(np.argsort(Lz)),:]
    Lz=np.flipud(np.sort(Lz))
    
    Rz=dR[:,0];
    R=R[np.argsort(Rz),:]
    Rz=np.flipud(np.sort(Rz))
    
    return L,R,X,Lz,Rz
    
def vizLRX(Lf,Rf,Xf,Lz,Rz,plotphase=1):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig=plt.figure(facecolor=(1,1,1))
    
    if (np.sum(np.angle(Lf[:,-1])!=0)+np.sum(np.angle(Rf[:,-1])!=0)+np.sum(np.angle(Xf[:,-1])!=0))==0:
        plotphase=0
    
    if plotphase==1:
        nrow=2
        fh=950
    else:
        #nrow=1    
        #fh=440
        nrow=2
        fh=950
        
    plt.get_current_fig_manager().window.setGeometry(120,85,1542,fh)
    
    L=np.real(Lf[:,:2])
    R=np.real(Rf[:,:2])
    X=np.real(Xf[:,:3])
    
    ####----------------------- MAGNITUDE
    Lx=np.absolute(Lf[:,-1])
    Rx=np.absolute(Rf[:,-1])
    Xm=np.absolute(Xf[:,-1])
    
    vmin=np.min(np.concatenate((Lx,Rx,Xm)));
    vmax=np.max(np.concatenate((Lx,Rx,Xm)));    
    
    cmap=plt.cm.get_cmap('viridis')
    #cmap=plt.cm.get_cmap('Reds')
    c_list=Lx
    colorL=cmap((c_list-vmin)/(vmax-vmin))
    i_list=np.ones(Lx.shape)
    i_list[Lx==0.0]=0.0
    colorL[:,-1]=i_list
    
    fig.add_subplot('{:}31'.format(nrow))
    plt.scatter(-L[:,0],L[:,1],c=colorL,linewidth=0,s=9,vmin=vmin,vmax=vmax,cmap='seismic')
    plt.title('Left Cortex')
    plt.ylabel('MAGNITUDE')
    plt.xlim([-80,120]);plt.ylim([-60,100])
    plt.xticks([]);plt.yticks([])  

    c_list=Rx
    colorR=cmap((c_list-vmin)/(vmax-vmin))
    i_list=np.ones(Rx.shape)
    i_list[Rx==0.0]=0.0
    colorR[:,-1]=i_list    
    
    fig.add_subplot('{:}32'.format(nrow))
    plt.scatter(R[:,0],R[:,1],c=colorR,linewidth=0,s=9,vmax=vmax,vmin=vmin,cmap='seismic')
    plt.title('Right Cortex')
    plt.xlim([-120,80]);plt.ylim([-60,100])
    plt.xticks([]);plt.yticks([])

    #cmap=plt.cm.get_cmap('seismic')
    c_list=Xm
    colors=cmap((c_list-vmin)/(vmax-vmin))
    i_list=np.array(c_list/vmax)**4.0
    colors[:,-1]=i_list
    
    ax3=fig.add_subplot('{:}33'.format(nrow),projection='3d');
    ax3.scatter(X[:,0],X[:,1],X[:,2],linewidth=0,c=colors);
    plt.axis('square')
    plt.title('Subcortical')
    #plt.colorbar()
    
    if plotphase==1:
        ####----------------------- PHASE
        Lx=np.angle(Lf[:,-1])
        Rx=np.angle(Rf[:,-1])
        Xx=np.angle(Xf[:,-1])
        
        vmin=-np.pi;
        vmax=np.pi;
        
        fig.add_subplot(234)
        plt.scatter(-L[:,0],L[:,1],c=Lx,linewidth=0,s=9,vmin=vmin,vmax=vmax,cmap='hsv')
        plt.title('Left Cortex')
        plt.ylabel('PHASE')
        plt.xlim([-80,120]);plt.ylim([-60,100])
        plt.xticks([]);plt.yticks([])  
        
        fig.add_subplot(235)
        plt.scatter(R[:,0],R[:,1],c=Rx,linewidth=0,s=9,vmax=vmax,vmin=vmin,cmap='hsv')
        plt.title('Right Cortex')
        plt.xlim([-120,80]);plt.ylim([-60,100])
        plt.xticks([]);plt.yticks([]) 
        
        cmap=plt.cm.hsv
        c_list=Xx
        colors=cmap((c_list-vmin)/(2*vmax))
        i_list=(c_list-vmin)/(2*vmax)
        colors[:,-1]=i_list
        
        ax3p=fig.add_subplot(2,3,6,projection='3d');
        ax3p.scatter(X[:,0],X[:,1],X[:,2],linewidth=0,c=colors,s=7);
        plt.axis('square')
        plt.title('Subcortical')
        plt.colorbar()
    else:
        ####----------------------- MAGNITUDE (FLIPPED)
    
        #Flipped Left Cortex
        Lflip=np.flipud(L)
        c_list=np.flipud(Lx)
        colorL=cmap((c_list-vmin)/(vmax-vmin))
        i_list=np.ones(Lx.shape)
        i_list[c_list==0.0]=0.0
        colorL[:,-1]=i_list
        colorL[np.flipud(Lz)>-30,:]=np.array([0.2,0.2,0.2,1])
        fig.add_subplot('234')
        plt.scatter(35+Lflip[:,0],Lflip[:,1],c=colorL,s=9,linewidth=0,vmin=vmin,vmax=vmax,cmap='seismic')
        plt.title('Left Cortex (Medial)')
        plt.ylabel('MAGNITUDE')
        plt.xlim([-80,120]);plt.ylim([-60,100])
        plt.xticks([]);plt.yticks([])  

        #Flipped Right Cortex
        
        Rflip=np.flipud(R)
        c_list=np.flipud(Rx)
        colorR=cmap((c_list-vmin)/(vmax-vmin))
        i_list=np.ones(Rx.shape)
        i_list[c_list==0.0]=0.0
        colorR[:,-1]=i_list
        colorR[np.flipud(Rz)>40,:]=np.array([0.2,0.2,0.2,1])
        fig.add_subplot('235')
        plt.scatter(10-Rflip[:,0],Rflip[:,1],c=colorR,s=9,linewidth=0,vmin=vmin,vmax=vmax,cmap='seismic')
        plt.title('Right Cortex (Medial)')
        plt.ylabel('MAGNITUDE')
        plt.xlim([-80,120]);plt.ylim([-60,100])
        plt.xticks([]);plt.yticks([])
        
        ax3.set_position([0.65,0.4,0.3,0.6])
        _,_,_,scb,scl=parseCoords()
        ax4=fig.add_subplot('236')
        violins=[]
        for n in range(len(scb)-1):
            #ax4.plot(n,np.mean(Xm[scb[n]:scb[n+1]]),'.')
            violins.append(ax4.violinplot(Xm[scb[n]:scb[n+1]],positions=[n],showmedians=True,showmeans=True,showextrema=True,bw_method='silverman'))
        for v in violins:
            v['cbars'].set_edgecolor([0.4,0.4,1])
            v['cmins'].set_edgecolor([0.4,0.4,1])
            v['cmedians'].set_edgecolor([0.4,0.4,1])
            v['cmaxes'].set_edgecolor([0.4,0.4,1])
            for vd in v['bodies']:
                vd.set_facecolor([0,0,1])
                vd.set_edgecolor('black')
        ax4.set_position([0.675,0.2,0.275,0.2])
        plt.xlim([-0.5,18.5])
        plt.ylim([vmin,vmax])
        plt.xticks(range(len(scb)-1),scl,rotation=-90)
        
        
    
   
    
#def plotGrayordMode(phi,plotphase=1,vmin=[],vmax=[],cmap='seismic',thresh=0,zeroout=False):
def plotGrayordMode(phi,plotphase=1):
    L,R,X,Lz,Rz=grayord2surf(phi);
    Lz=np.flipud(Lz);Rz=np.flipud(Rz)
    vizLRX(L,R,X,Lz,Rz,plotphase=plotphase)
        
def loadGrayord(fname,norm=True):
    img=nib.load(fname);data=img.get_data()
    data=data[0,0,0,0,:,:].T
    if norm:
        data=(data-np.mean(data,1)[:,None])/np.std(data,1)[:,None]
    return data
    
if __name__=='__main__':
    pass