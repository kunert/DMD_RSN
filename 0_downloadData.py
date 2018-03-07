# -*- coding: utf-8 -*-
"""
Created on 3/3/17 by JK-G
Run this script to auto-download HCP data from the Amazon s3 server

Note that this requires a username and keys, which can be acquired from the HCP. For more info see:
https://wiki.humanconnectome.org/display/PublicData/Using+ConnectomeDB+data+on+Amazon+S3

This script will prompt for a username, key, and secret key, then save that info in ./data/s3keys.h5
(please be mindful to keep this file secure)


NOTE:
If you already have the data locally it will not be necessary to run this script.
The included notebooks have a string 'datapath' which may simply be pointed to where the data exists
Please be sure that the data is organized as follows:
if datapath='/example/folder/data/'
files are in folders by subject number, e.g.
/example/folder/data/102513/rfMRI_REST1_LR_Atlas.dtseries.nii
"""

import numpy as np
import boto3
import os
import time
import h5py

#define matlab-like tictoc functions
def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()
def toc():
    print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."


#make a data subfolder, if it doesn't already exist
try:
    os.mkdir('./data')
except:
    pass

if os.path.exists('./s3keys.h5'):
    with h5py.File('./s3keys.h5','r') as hf:
        user=hf['user'].value
        key=hf['key'].value
        skey=hf['skey'].value
else:
    print 's3key credentials file not found! Please enter HCP username, key and secret key:'
    user=raw_input("user:")
    key=raw_input('key:')
    skey=raw_input('skey:')
    with h5py.File('./s3keys.h5','w') as hf:
        hf.create_dataset('user',data=user)
        hf.create_dataset('key',data=key)
        hf.create_dataset('skey',data=skey)
    

#open boto3 session to access hcp data
boto3.setup_default_session(profile_name='hcp')
s3=boto3.resource('s3')
bucket=s3.Bucket('hcp-openaccess')

#load u120 list
u120=np.genfromtxt('./u120.txt').astype(int).astype(str)
ulist=u120

for subject in ulist:
    #make a folder for the subject if it doesn't already exist
    folder='./data/'+subject+'/'
    try:
        os.mkdir(folder)
    except:
        pass

    #loop through and download resting state data
    for task in ['REST1','REST2']:
        for trialdir in ['LR','RL']:
            trial='rfMRI_'+task+'_'+trialdir
            print subject+':'+trial
            #try to download the corresponding scan data if it doesn't already exist
            #but terminate the loop if the user ctrl-c's out
            #or, if something else fails along the way, terminate with an error message...
            try:
                fname='HCP_900/'+subject+'/MNINonLinear/Results/'+trial+'/'+trial+'_Atlas.dtseries.nii'
                if os.path.exists(folder+trial+'_Atlas.dtseries.nii') is False:
                    tic()
                    bucket.download_file(fname,folder+trial+'_Atlas.dtseries.nii')
                    toc()
                else:
                    print "File already downloaded! Moving on..."
            except (KeyboardInterrupt,SystemExit):
                    raise
            except:
                print 'DOWNLOAD FAILED'