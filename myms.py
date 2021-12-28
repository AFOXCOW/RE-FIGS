# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pymzml
from bidict import bidict
from scipy import sparse
import struct
import zlib
import sqlite3
import sparse_nnls

def LoadMS2(path):
    DIArun = pymzml.run.Reader(path)
    E = enumerate(DIArun)
    MS2 = [[spectrum.peaks('raw'),(spectrum['MS:1000827']-spectrum['MS:1000828'], spectrum['MS:1000827']+spectrum['MS:1000829']),
            spectrum['MS:1000016'],i+1] for i,spectrum in E if spectrum['ms level']==2.0]
    return MS2

def GenerateDecoyLibrary(SpectraLibrary, distance):
    DecoyLibrary = {}
    charges = set(np.array(list(SpectraLibrary.keys()))[:,1].astype(int))
    for charge in charges:
        LibraryDividedByCharge = {key:value for key, value in SpectraLibrary.items() if key[1]==charge}
        print('Charge: '+str(charge))
        DecoyLibraryDividedByCharge = {}
        swapped_keys = bidict()

        key_premz = []
        for key, value in LibraryDividedByCharge.items():
            key_premz.append([key, value['PrecursorMZ']])
        key_premz = np.array(key_premz, dtype=object)
        key_premz = key_premz[np.argsort(key_premz[:,1])]

        i, j = 0, 1
        while i < len(key_premz) and j < len(key_premz):
            if i%1000==0:
                print('\r' + str(i)+"/"+str(len(LibraryDividedByCharge)), end='', flush=True)
            i_key, i_premz, j_key, j_premz = key_premz[i][0], key_premz[i][1], key_premz[j][0], key_premz[j][1]
            if abs(j_premz-i_premz) >= distance:
                decoy_i_key = ('DECOY-'+i_key[0], i_key[1])
                decoy_j_key = ('DECOY-'+j_key[0], j_key[1])
                
                DecoyLibraryDividedByCharge[decoy_i_key] = SpectraLibrary[i_key].copy()
                DecoyLibraryDividedByCharge[decoy_i_key]['Spectrum'] = SpectraLibrary[j_key]['Spectrum'].copy()

                DecoyLibraryDividedByCharge[decoy_j_key] = SpectraLibrary[j_key].copy()
                DecoyLibraryDividedByCharge[decoy_j_key]['Spectrum'] = SpectraLibrary[i_key]['Spectrum'].copy()
                
                delta = i_premz - j_premz
                DecoyLibraryDividedByCharge[decoy_i_key]['Spectrum'][:,0] += delta
                DecoyLibraryDividedByCharge[decoy_j_key]['Spectrum'][:,0] -= delta

                # filter peaks with mz<=0
                DecoyLibraryDividedByCharge[decoy_i_key]['Spectrum'] = DecoyLibraryDividedByCharge[decoy_i_key]['Spectrum'][DecoyLibraryDividedByCharge[decoy_i_key]['Spectrum'][:,0]>0]
                DecoyLibraryDividedByCharge[decoy_j_key]['Spectrum'] = DecoyLibraryDividedByCharge[decoy_j_key]['Spectrum'][DecoyLibraryDividedByCharge[decoy_j_key]['Spectrum'][:,0]>0]
                ###################

                swapped_keys[i_key] = j_key
            else:
                j += 1

            while (i < len(key_premz)) and (key_premz[i][0] in (set(swapped_keys) | set(swapped_keys.inv))):
                i += 1
            while (j < len(key_premz)) and (key_premz[j][0] in (set(swapped_keys) | set(swapped_keys.inv))):
                j += 1
        print('')
        if len(LibraryDividedByCharge) != len(DecoyLibraryDividedByCharge):
            unswapped_keys = set(LibraryDividedByCharge.keys()) - (set(swapped_keys) | set(swapped_keys.inv))
            unswapped_keys = list(unswapped_keys)
            unswapped_keys.sort()

            for unswapped_key in unswapped_keys:
                unswapped_premz = SpectraLibrary[unswapped_key]['PrecursorMZ']
                for swapped_key in sorted(swapped_keys):
                    swapped_key2 = swapped_keys[swapped_key]
                    if abs(unswapped_premz - SpectraLibrary[swapped_key]['PrecursorMZ'])>= distance and \
                       abs(unswapped_premz - SpectraLibrary[swapped_key2]['PrecursorMZ'])>= distance:
                        DecoyLibraryDividedByCharge[('DECOY-'+unswapped_key[0], unswapped_key[1])] = SpectraLibrary[unswapped_key].copy()
                        DecoyLibraryDividedByCharge[('DECOY-'+unswapped_key[0], unswapped_key[1])]['Spectrum'] = SpectraLibrary[swapped_key2]['Spectrum'].copy()

                        DecoyLibraryDividedByCharge[('DECOY-'+swapped_key[0], swapped_key[1])]['Spectrum'] = SpectraLibrary[unswapped_key]['Spectrum'].copy()
                        
                        DecoyLibraryDividedByCharge[('DECOY-'+swapped_key[0], swapped_key[1])]['Spectrum'][:,0] += (SpectraLibrary[swapped_key]['PrecursorMZ']-unswapped_premz)
                        DecoyLibraryDividedByCharge[('DECOY-'+unswapped_key[0], unswapped_key[1])]['Spectrum'][:,0] += (SpectraLibrary[unswapped_key]['PrecursorMZ']-SpectraLibrary[swapped_key2]['PrecursorMZ'])
                        
                        # filter peaks with mz<=0
                        DecoyLibraryDividedByCharge[('DECOY-'+swapped_key[0], swapped_key[1])]['Spectrum'] = DecoyLibraryDividedByCharge[('DECOY-'+swapped_key[0], swapped_key[1])]['Spectrum'][DecoyLibraryDividedByCharge[('DECOY-'+swapped_key[0], swapped_key[1])]['Spectrum'][:,0]>0]
                        DecoyLibraryDividedByCharge[('DECOY-'+unswapped_key[0], unswapped_key[1])]['Spectrum'] = DecoyLibraryDividedByCharge[('DECOY-'+unswapped_key[0], unswapped_key[1])]['Spectrum'][DecoyLibraryDividedByCharge[('DECOY-'+unswapped_key[0], unswapped_key[1])]['Spectrum'][:,0]>0]
                        ###################     
                        
                        break
                if swapped_key in swapped_keys: swapped_keys.pop(swapped_key)
        DecoyLibrary.update(DecoyLibraryDividedByCharge)
    return DecoyLibrary

def cal_nnls(LibIntensity, MS2Intensity, penalty):
    RowIndex = list(range(len(LibIntensity) + 1))
    ColIndex = [0] * (len(LibIntensity) + 1)
    LibIntensity.append(penalty)

    MS2Intensity.append(0)
    MS2Intensity = np.array(MS2Intensity)

    LibraryVector = sparse.coo_matrix((LibIntensity, (RowIndex, ColIndex)))
    LibraryCoeffs = sparse_nnls.lsqnonneg(LibraryVector, MS2Intensity, {'show_progress': False})
    LibraryCoeffs = LibraryCoeffs['x']
    LibraryCoeffs = LibraryCoeffs[0]

    return LibraryCoeffs


def Peaks_match(lib_spc, exp_spc, tol=2e-5):
    i=0
    j=0
    tmp = []
    match_mz = []
    match_lib_it = []
    match_exp_it = []
    while i!=len(exp_spc):
        peak_i_mz = exp_spc[i][0]
        peak_i_it = exp_spc[i][1]
        while j!=len(lib_spc):
            peak_j_mz = lib_spc[j][0]
            peak_j_it = lib_spc[j][1]
            if peak_i_mz*(1-tol)<=peak_j_mz<=peak_i_mz*(1+tol):
                match_mz.append(peak_j_mz)
                match_lib_it.append(peak_j_it)
                match_exp_it.append(peak_i_it)
                i+=1
                j+=1
                break
            elif peak_i_mz*(1-tol)>peak_j_mz:
                j+=1
                break
            else:
                i+=1
                break
        if j==len(lib_spc):
            break
    if len(match_mz)==0:
        print('No matching peaks!')

    match_lib_it = np.array(match_lib_it)
    match_exp_it = np.array(match_exp_it)
    return match_mz,match_lib_it,match_exp_it




def deconvolute(SpectrumInfo, SpectraLibrary, tol, Top10First, level=1):
    SpectrumInfo = copy.deepcopy(SpectrumInfo)
    window = SpectrumInfo[1]
    windowMZ = (window[0] + window[1]) / 2.0
    spectrumRT = SpectrumInfo[2]
    index = SpectrumInfo[3]
    DIASpectrum = np.array(SpectrumInfo[0])
    if DIASpectrum.shape[0] == 0:
        return [[0, index, 0, 0, windowMZ, spectrumRT, level, 0]]
    RefSpectraLibrary = SpectraLibrary.copy()

    PrecursorCandidates = list(RefSpectraLibrary.keys())
#     if 'PrecursorRT' in list(SpectraLibrary.values())[0]:
#         PrecursorCandidates = [key for key,spectrum in RefSpectraLibrary.items()
#                                 if spectrumRT-5 < float(spectrum['PrecursorRT']) < spectrumRT+5]

    CandidateLibrarySpectra = [RefSpectraLibrary[key]['Spectrum'] for key in PrecursorCandidates]

    peaknum = len(DIASpectrum)
    while True:
        MergedSpectrumCoordIndices = np.searchsorted(DIASpectrum[:, 0] + tol * DIASpectrum[:, 0], DIASpectrum[:, 0])
        MergedSpectrumCoords = DIASpectrum[np.unique(MergedSpectrumCoordIndices), 0]
        MergedSpectrumIntensities = [np.mean(DIASpectrum[np.where(MergedSpectrumCoordIndices == i)[0], 1]) for i in
                                    np.unique(MergedSpectrumCoordIndices)]
        DIASpectrum = np.array((MergedSpectrumCoords, MergedSpectrumIntensities)).transpose()
        if peaknum == len(DIASpectrum):
            break
        peaknum = len(DIASpectrum)

    CriticalMZs = np.concatenate(
        (DIASpectrum[:, 0] - tol * DIASpectrum[:, 0], DIASpectrum[:, 0] + tol * DIASpectrum[:, 0]))
    CriticalMZs = np.sort(CriticalMZs)

    LocateReferenceCoordsInDIA = [np.searchsorted(CriticalMZs, lib[:, 0]) for lib in CandidateLibrarySpectra]

    TopTenPeaksCoordsInDIA = [np.searchsorted(CriticalMZs, M[np.argsort(-M[:, 1])[0:min(10, M.shape[0])], 0]) for M
                              in CandidateLibrarySpectra]

    ReferencePeaksInDIA = [i for i in range(len(PrecursorCandidates)) if
                           len([a for a in TopTenPeaksCoordsInDIA[i] if a % 2 == 1]) >= 5
                           ]

    IdentPrecursorsLocations = [LocateReferenceCoordsInDIA[i] for i in ReferencePeaksInDIA]

    IdentPrecursorSpectra = [CandidateLibrarySpectra[i] for i in ReferencePeaksInDIA]

    IdentPrecursors = [PrecursorCandidates[i] for i in ReferencePeaksInDIA]
    if len(IdentPrecursors) == 0:
        return [[0, index, 0, 0, windowMZ, spectrumRT, level, 0]]

    Penalties = np.array([
        np.sum([
            IdentPrecursorSpectra[j][k][1]
            for k in range(len(IdentPrecursorSpectra[j]))
            if IdentPrecursorsLocations[j][k] % 2 == 0
        ]) for j in range(len(IdentPrecursorSpectra))
    ])

    RowIndices = (np.array([i for v in IdentPrecursorsLocations
                                         for i in v if i % 2 == 1])
                               + 1) / 2
    RowIndices = RowIndices - 1
    RowIndices = RowIndices.astype(int)

    ColumnIndices = np.array([
        i for j in range(len(IdentPrecursors))
        for i in [j] * len([k for k in IdentPrecursorsLocations[j] if k % 2 == 1])
    ])

    MatrixIntensities = np.array([
        IdentPrecursorSpectra[k][i][1]
        for k in range(len(IdentPrecursorSpectra))
        for i in range(len(IdentPrecursorSpectra[k]))
        if IdentPrecursorsLocations[k][i] % 2 == 1
    ])

    UniqueRowIndices = list(set(RowIndices))
    UniqueRowIndices.sort()
    DIASpectrumIntensities = DIASpectrum[UniqueRowIndices, 1]

    relatedLibNums = Counter(RowIndices)
    counter_keys = np.array(list(relatedLibNums.keys()))
    counter_values = np.array(list(relatedLibNums.values()))

    unique_mz_ids = counter_keys[np.where(counter_values == 1)]
    unique_mz_lib_ids = np.array(
        [ColumnIndices[np.where(RowIndices == mz_id)] for mz_id in unique_mz_ids])
    unique_mz_lib_ids = unique_mz_lib_ids.reshape(1, -1)[0]

    flag = False
    output = []
    for i in range(len(IdentPrecursors)):
        coeff = 0
        unique_mz_ids_of_thislib = unique_mz_ids[np.where(unique_mz_lib_ids == i)]
        if len(unique_mz_ids_of_thislib) > 0:
            FeaturedPeakIntensitiesInDIA = DIASpectrum[unique_mz_ids_of_thislib, 1]
            FeaturedPeakIntensitiesInLib = np.array(
                [MatrixIntensities[np.where(RowIndices == mz_id)] for mz_id in unique_mz_ids_of_thislib])
            FeaturedPeakIntensitiesInLib = FeaturedPeakIntensitiesInLib.reshape(1, -1)[0]

            if Top10First:
                CandidateIndex = np.where(FeaturedPeakIntensitiesInLib >= MinIntensityDic[IdentPrecursors[i]])
                if len(CandidateIndex[0]) > 0:
                    FeaturedPeakIntensitiesInDIA = FeaturedPeakIntensitiesInDIA[CandidateIndex]
                    FeaturedPeakIntensitiesInLib = FeaturedPeakIntensitiesInLib[CandidateIndex]

            coeff = cal_nnls(list(FeaturedPeakIntensitiesInLib), list(FeaturedPeakIntensitiesInDIA), Penalties[i])
        else:
            flag = True
        output.append(
            [coeff, index, IdentPrecursors[i][0], IdentPrecursors[i][1], windowMZ, spectrumRT, level])

    RowIndices = stats.rankdata(RowIndices, method='dense').astype(int) - 1
    if flag and level < 50:
        LibraryMatrix = sparse.coo_matrix((MatrixIntensities, (RowIndices, ColumnIndices)))
        LibraryCoeffs = [i[0] for i in output]
        FittingIntensities = LibraryMatrix * LibraryCoeffs
        FittingIntensities = FittingIntensities.reshape(1, -1)[0]
        DIASpectrum[UniqueRowIndices, 1] = DIASpectrum[UniqueRowIndices, 1] - FittingIntensities

        DIASpectrum[:, 1][np.where(DIASpectrum[:, 1] < 0)] = 0

        output = [out for out in output if out[0] > 0]
        CopyLibrary = SpectraLibrary.copy()
        for out in output:
            CopyLibrary.pop((out[2], out[3]))

        SpectrumInfo[0] = DIASpectrum
        output.extend(deconvolute(SpectrumInfo, CopyLibrary, tol, Top10First, level + 1))

    if level == 1:
        LibraryMatrix = sparse.coo_matrix((MatrixIntensities, (RowIndices, ColumnIndices)))
        LibraryCoeffs = []
        for lib_id in range(len(IdentPrecursors)):
            for out in output:
                if (IdentPrecursors[lib_id] == (out[2], out[3])):
                    LibraryCoeffs.append(out[0])
        FittingIntensities = LibraryMatrix * LibraryCoeffs
        FittingIntensities = FittingIntensities.reshape(1, -1)[0]

        Correlation = pd.DataFrame([FittingIntensities, DIASpectrumIntensities]).T.corr()[0][1]
        for i in output:
            i.append(Correlation)

    return output

