## Resting State Networks using Dynamic Mode Decomposition

Scripts to accompany the manuscript "Extracting Reproducible Time-Resolved Resting State Networks using Dynamic Mode Decomposition". This repo includes jupyter notebooks which implement the methods for generating figures as in the paper, starting directly from the Human Connectome Project data.

The scripts and notebooks are described below:

---
### 0_downloadData.py
Unlike the rest, this is a python script as opposed to a jupyter notebook. Its purpose is to download the resting-state HCP data for the individuals analyzed in the paper. It does so by downloading directly from the HCP's Amazon s3 server, and requires credentials for accessing this. These credentials must be acquired from the HCP as described here: 

https://wiki.humanconnectome.org/display/PublicData/Using+ConnectomeDB+data+on+Amazon+S3

The script will ask for your s3 keys when run.

** If you already have the HCP data, you do not need to run this script! ** Each notebook includes a variable which points to the filepath of the data, and you can simply adjust this to match where your data is stored. This is important to note, because the amount of data is quite large: each scan is 439MB, and we analyze 120 individuals who each have 4 scans, for a total size of almost 210GB. 

---
### 1_Calculate_DMD.ipynb
This notebook iterates through a set of HCP scans and calculates windowed DMD modes.

This typically takes 26 seconds per scan on my machine, and generates about 40.2MB of output per scan when using the default parameters. For the set of scans used in the paper (120 individuals, 4 scans each) this takes about 3.5 hours and generates 19.7GB of saved output.

---
### 2_Calculate_Clusters-gDMD.ipynb / 2_Calculate_Clusters-sDMD.ipynb
These notebooks implement gDMD/sDMD clustering, generating figures resembling **Figure 2** and **Figure 3**, respectively.

Execution time depends mostly on the clustering steps, which depends strongly on the number of modes being clustered. This ranges from a few minutes for an sDMD run, up to a few hours for a large gDMD run. In both cases, the notebook outputs and saves the cluster labels and information for the set of input modes, which will be on the order of a few hundred kb.


---
### 3_sDMDvGICA.ipynb
This script runs the analysis which compares traditional ICA to our sDMD approach in generating subject-level Default Mode Networks, using gICA+Dual Regression DMNs as a baseline. This generates a figure resembling **Figure 4**.

This process involves running both sDMD and ICA over a range of parameters, with each method taking a variable amount of time. sDMD ranged from 3-10 minutes, and the ICA analysis takes approximately ten minutes per subject. I left this to run overnight during which time it completed.
For every individual, this script generates and saves the best parameters/modes for each method along with a figure showing the best sDMD/ICA modes alongside the gICA+DR mode (a total of 230kb per individual, for a total 32MB of output).

**Important Note**: Before running this script one must download the gICA+Dual Regression results from the HCP. This is another large download, and the notebook contains instructions on how to do this.

---
### 4_Characterize_Dynamics.ipynb
This notebook analyzes the dynamics of the modes within each cluster, generating figures resembling **Figure 5** and **Figure 6**. Execution time on my machine is a few minutes, and no output is generated besides these figures.

**Important Note**: This notebook assumes that **2_Calculate_Clusters-gDMD.ipynb** has been used to analyze all four sets of scans using the default parameters. See the notebook for more information.
