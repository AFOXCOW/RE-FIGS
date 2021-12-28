import os 
import sys
import numpy as np
import pandas as pd
from scipy import stats, sparse
from collections import Counter
import pickle
from tqdm import tqdm
import copy 
import sys
sys.path.append('../..')
from myms import *

parameters = sys.argv
mzML_file = parameters[1]
SSM_file = parameters[2]
lib_file = parameters[3]
topN = int(parameters[4])
tol = float(parameters[5]) * 1e-6 #ppm


with open(lib_file,'rb')as f:
    library = pickle.load(f)

for key in library.keys():
    spc = library[key]['Spectrum']
    ascending = np.argsort(spc[:,1])
    new_spc = spc[ascending[::-1]][:topN]
    new_ascending = np.argsort(new_spc[:,0])
    new_spc = new_spc[new_ascending]
    new_spc[:,1] = new_spc[:,1]/np.sum(new_spc[:,1])
    library[key]['Spectrum'] = new_spc


Identify = pd.read_csv(SSM_file)
Identify['Precursor'] = Identify[['peptide','zLIB']].apply(lambda x:x['peptide']+'_'+str(x['zLIB']),axis=1)
MS2 = LoadMS2(mzML_file)
MS2_dict = {}
for ms2 in MS2:
    MS2_dict[ms2[-1]] = ms2

output = []
scans = set(Identify['scan'].values)
for scan in scans:
    tmp_df = Identify[Identify['scan']==scan]
    keys = tmp_df['Precursor'].values
    tmp_library = {}
    for key in keys:
        tmp_library[(key.split('_')[0],int(key.split('_')[1]))] = library[(key.split('_')[0],int(key.split('_')[1]))]
    tmp_ms2 = MS2_dict[scan]
    output.extend(deconvolute(tmp_ms2,tmp_library,tol,False))

Coeffs = pd.DataFrame(output)
Coeffs.to_csv(SSM_file.replace('.csv','_Coeffs.csv'), index=False, header=['Coeff','index','peptide','charge','windowMZ','spectrumRT','level','corr'])
print('over')