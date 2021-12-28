#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import math
import time 
import random
import sys
sys.path.append('../..')
from myms import *
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

parameters = sys.argv
mzML_file = parameters[1]
SSM_file = parameters[2]
lib_file = parameters[3]
start_cycle = int(parameters[4])
end_cycle = int(parameters[5]) # not include

good_shared_limit = int(parameters[6])
good_cos_sim_limit = float(parameters[7])
good_sqrt_cos_sim_limit = float(parameters[8])
good_count_within_cycle_limit = int(parameters[9])

if len(parameters)>10:
    tol = float(parameters[10])*1e-6
else:
    tol = 30*1e-6

if len(parameters)>11:
    scans_per_cycle = int(parameters[11])
else:
    scans_per_cycle = 1500

if len(parameters)>12:
    seed = int(parameters[12])
else:
    seed = 0
    
cycle_range = range(start_cycle,end_cycle)
cycle_number = len(cycle_range)

################################
# Spectral Library
################################
with open(lib_file,'rb')as f:
    library = pickle.load(f)
# only  keep top10 peaks
topN=10
for key in (library.keys()):
    spc = library[key]['Spectrum']
    ascending = np.argsort(spc[:,1])
    new_spc = spc[ascending[::-1]][:topN]

    new_ascending = np.argsort(new_spc[:,0])
    new_spc = new_spc[new_ascending]
    
    # peaks intensity normalization
    new_spc[:,1] = new_spc[:,1]/np.sum(new_spc[:,1])
    library[key]['Spectrum'] = new_spc


print("Loading SSM results!")
df = pd.read_csv(SSM_file)
if not 'cycle' in df.columns:
    df['cycle'] = df['scan'].apply(lambda x:(x-1)//scans_per_cycle+1)

# cycle filtering
df = df[df['cycle'].isin(cycle_range)]
df = df.reset_index()
df['Precursor'] = df[['peptide','zLIB']].apply(lambda x:x['peptide']+'_'+str(x['zLIB']),axis=1)

#############################################
# Loading MS2 and change list to dictionary
#############################################
print("Loading MS2!")
MS2 = LoadMS2(mzML_file)
MS2_dict = {}
for ms2 in (MS2):
    MS2_dict[ms2[-1]] = ms2[:-1]


###################################
# Feature extraction (calculation)
###################################
match_peaks = pd.DataFrame([])
scan_list = df['scan'].values
peptide_list = df['peptide'].values
zLIB_list = df['zLIB'].values
cosine_list = df['cosine'].values
Peaks_Library_list = df['Peaks(Library)'].values
shared_list = df['shared'].values
MaCC_list = df['MaCC_Score'].values
cycle_list = df['cycle'].values

cos_sim_list = np.zeros(len(df))
sqrt_cos_sim_list = np.zeros(len(df))
norm_mse_list = np.zeros(len(df)) 
norm_mae_list = np.zeros(len(df))
match_it_ratio_list = np.zeros(len(df))
match_number_list = np.zeros(len(df))
print("Calculating features!")
for idx,row in df.iterrows():
    scan = row['scan']
    precursor = row['Precursor']
    key = (row['peptide'],row['zLIB'])
    lib_spectrum = library[key]
    exp_spectrum = MS2_dict[scan]
    lib_spc = lib_spectrum['Spectrum']
    exp_spc = exp_spectrum[0]

    match_mz,match_lib_it, match_exp_it = Peaks_match(lib_spc, exp_spc, tol)

    match_lib_it = np.array(match_lib_it)
    match_exp_it = np.array(match_exp_it)
    cos_sim = cosine_similarity(match_lib_it.reshape(1,len(match_lib_it)),match_exp_it.reshape(1,len(match_exp_it)))[0][0]

    sqrt_match_lib_it = np.sqrt(match_lib_it)
    sqrt_match_exp_it = np.sqrt(match_exp_it)
    sqrt_cos_sim = cosine_similarity(sqrt_match_lib_it.reshape(1,len(sqrt_match_lib_it)),sqrt_match_exp_it.reshape(1,len(sqrt_match_exp_it)))[0][0]

    l2_norm_sqrt_match_lib_it = match_lib_it/np.linalg.norm(match_lib_it)
    l2_norm_sqrt_match_exp_it = match_exp_it/np.linalg.norm(match_exp_it)
    norm_mse = mean_squared_error(l2_norm_sqrt_match_lib_it,l2_norm_sqrt_match_exp_it)
    norm_mae = np.mean(abs(l2_norm_sqrt_match_lib_it-l2_norm_sqrt_match_exp_it))

    match_it_ratio = np.sum(match_lib_it)/np.sum(lib_spc[:,1])

    cos_sim_list[idx] = cos_sim
    sqrt_cos_sim_list[idx] = sqrt_cos_sim
    norm_mse_list[idx] = norm_mse
    norm_mae_list[idx] = norm_mae
    match_it_ratio_list[idx] = match_it_ratio
    match_number_list[idx] = int(len(match_lib_it))

match_peaks['scan'] = scan_list
match_peaks['peptide'] = peptide_list
match_peaks['zLIB'] = zLIB_list
match_peaks['cosine'] = cosine_list
match_peaks['Peaks(Library)'] = Peaks_Library_list
match_peaks['shared'] = shared_list
match_peaks['MaCC_Score'] = MaCC_list
match_peaks['cycle'] = cycle_list
match_peaks['cos_sim'] = cos_sim_list
match_peaks['sqrt_cos_sim'] = sqrt_cos_sim_list
match_peaks['norm_mse'] = norm_mse_list
match_peaks['norm_mae'] = norm_mae_list
match_peaks['match_it_ratio'] = match_it_ratio_list
match_peaks['match_number'] = match_number_list
feature_filepath = SSM_file.replace('.csv','_withFeature_'+str(cycle_number)+'cycle.csv')
match_peaks.to_csv(feature_filepath,index=False)



################################
# RI-FIGS distinguish decoy by the peptide sequence.
# Decoy starts with "DECOY-"
################################
df = pd.read_csv(feature_filepath)
df['protein'] = df['peptide'].apply(lambda x:'DECOY_null' if x.startswith('DECOY-') else 'TARGET')
df['label'] = df['protein'].apply(lambda x:0 if x=='DECOY_null' else 1)

cycle_max = max(df['cycle'].values)

# calculate the number of peptide within cycle. and keep peptide with best MaCC_Score
df_dup_rm = pd.DataFrame([])
for i in (range(1,cycle_max+1)):
    df_cycle = df[df['cycle']==i]
    count = df_cycle['peptide'].value_counts()
    count_df = count.reset_index()
    count_df['peptide'] = count.index
    count_df['count_within_cycle'] = count.values
    count_df = pd.DataFrame(count_df[['peptide','count_within_cycle']].values)
    count_df.columns = ['peptide','count_within_cycle']
    df_cycle = df_cycle.sort_values('MaCC_Score', ascending=False).groupby('peptide', as_index=False).first()
    df_cycle = pd.merge(df_cycle,count_df)
    df_dup_rm = df_dup_rm.append(df_cycle)
    

df_run = df_dup_rm.copy()
# calculate the number of peptide between cycle. and keep peptide with best MaCC_Score
count = df_run['peptide'].value_counts()
count_df = count.reset_index()
count_df['peptide'] = count.index
count_df['cycle_count'] = count.values
count_df = pd.DataFrame(count_df[['peptide','cycle_count']].values)
count_df.columns = ['peptide','cycle_count']
df_run = df_run.sort_values('MaCC_Score', ascending=False).groupby('peptide', as_index=False).first()
df_run = pd.merge(df_run,count_df)  



#########################
# seed for reproduction
#########################
np.random.seed(seed)
final_df = df_run.copy()
target = final_df[final_df['protein']=='TARGET']
decoy = final_df[final_df['protein']!='TARGET']


# choose good target peptide with higher requirements.
# the choosen good target can affect the classification.
good_target = target[target['shared']>good_shared_limit]
good_target = good_target[good_target['cos_sim']>good_cos_sim_limit]
good_target = good_target[good_target['sqrt_cos_sim']>good_sqrt_cos_sim_limit]
good_target = good_target[good_target['count_within_cycle']>=good_count_within_cycle_limit]
if len(good_target)<500:
    print("Not a warning, good target samples number is "+"%d"%len(good_target)+", less than 500!")

size = len(good_target)
weights_list = []
print("Ensemble learning LDA!")
# Ensemble learning。Train 10 LDA model and average the weights.
for j in range(10):
    randidx = np.random.randint(0,len(decoy),size)
    choosen_decoy = decoy.iloc[randidx]
    DataSet_df = good_target.append(choosen_decoy)
    col = ['shared', 'MaCC_Score','cos_sim','sqrt_cos_sim', 'norm_mse','norm_mae','match_it_ratio','count_within_cycle','cycle_count','label']
    DataSet = np.array(DataSet_df[col])
    X = DataSet[:, :-1]
    y = DataSet[:, -1].astype(int)
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    weights = clf.coef_[0]
    weights_list.append(weights)

weights_list = np.array(weights_list)
avg_weights = np.mean(weights_list,axis=0)
# calculate the final LDA score by weights the other features.
total_X = np.array(final_df[col])[:,:-1]
scores = np.dot(total_X, avg_weights)
final_df['LDA_Score'] = scores


# FDR calculate
final_df = final_df.sort_values(by="LDA_Score" , ascending=False)
FDR = []
decoy_number = 0
target_number = 0
for idx,row in final_df.iterrows():
    protein = row['protein']
    if protein=='DECOY_null':
        decoy_number+=1
    else:
        target_number+=1
    # for decoy/target in library is 1:1
    FDR.append(decoy_number/target_number)
final_df['FDR'] = FDR

# p-value calculate
final_df = final_df.sort_values(by="LDA_Score" , ascending=False)
pvalue = []
decoy_number = 0
total_decoy = len(decoy)
for idx,row in final_df.iterrows():
    protein = row['protein']
    if protein=='DECOY_null':
        decoy_number+=1
    else:
        target_number+=1
    # for decoy/target in library is 1:1
    pvalue.append((decoy_number+1)/(total_decoy+1))
final_df['p-value'] = pvalue
final_df.to_csv(feature_filepath.replace('.csv','_pvalue.csv'),index=False)

# Q-value filter.
# find the least LDA_score which FDR<=0.01    
cutoff = 1e9
reverse_df_dup_rm = final_df.iloc[::-1]
for idx,row in reverse_df_dup_rm.iterrows():
    if row['FDR']<=0.01:
        cutoff = row['LDA_Score']
        break
good = final_df[final_df['LDA_Score']>=cutoff]
good.to_csv(feature_filepath.replace('.csv','_LDA_ID.csv'),index=False)
print('finished')