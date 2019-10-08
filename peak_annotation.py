# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:10:34 2019

@author: qlj874

Usage:
python peak_annotation.py <path to combined dir> <match_tolerance>

This script categorizes ions from MS2 spectra and collects their intensities.
As input, the MaxQuant output folder is used. Specifically, the msms.txt table
and the peak files (.apl)
The peaks are categorized in
- precursor
- ET-no-D (charge reduced precursors)
- fragments (as identified by MaxQuant)
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from tqdm import tqdm

#combined_dir = 'C:/Users/qlj874/Documents/ETnoD/ET_noD_test/combined'
#combined_dir = 'C:/Users/qlj874/Documents/HeLa_NCE_tests/combined'
#match_tolerance = 0.05

combined_dir = sys.argv[1]
match_tolerance = float(sys.argv[2])


def match_float_to_array(x, search_array, tol=match_tolerance):
    """
    This function finds the closest match for an float to an array of floats
    and returns the match index from search_array if it is below a given
    tolerance (in Da) or -1 if it's above the tolerance.
    """
    
    diffs = np.absolute(search_array - x)
    min_diff = diffs.min()
    min_index = np.where(diffs == min_diff)
    if min_diff <= tol:
        if len(min_index[0]) != 1:
            # if there are multiple matches just take the first one but raise
            # a warning
            warnings.warn('Multiple matches found, taking the first one')
            min_index = min_index[0]
        return(int(min_index[0]))
    else:
        return(-1)

def main():
    
    print('read apl files')
    # scan folder for apl files
    apl_files = []
    for file in os.listdir(combined_dir + '/andromeda/'):
        if file.endswith(".apl") and not 'secpep' in file:
            apl_files.append(os.path.join(combined_dir + '/andromeda/', file))
    
    # parse the apl file into a list of dicts
    # this is a very slow parser that could be optimized
    spectra = []
    for apl_file in tqdm(apl_files):
        with open(apl_file, 'r') as f:
            for l in f.readlines():
                if l == 'peaklist start\n':
                    row = {}
                    mz = []
                    intens = []
                elif l[:2] == 'mz':
                    row['mz'] = float(l[3:].strip())
                elif l[:13] == 'fragmentation':
                    row['fragmentation'] = l[14:].strip()
                elif l[:6] == 'charge':
                    row['charge'] = int(l[7:].strip())
                elif l[:6] == 'header':
                    x = l.split(' ')
                    row['raw_file'] = x[1]
                    row['index'] = int(x[3])
                elif l == 'peaklist end\n':
                    row['mz_ar'] = np.array(mz)
                    row['int_ar'] = np.array(intens)
                    spectra.append(row)
                elif l == '\n':
                    continue
                else:
                    vals = [x.strip() for x in l.split('\t')]
                    mz.append(float(vals[0]))
                    intens.append(float(vals[1]))
    
    # for each spectrum, annotate precursor intensity and etnoD intensities.
    # since MaxQuant performs charge deconvolution on every isotope pattern
    # with the assumption that the charge state is purely dependent on added
    # protons, charge reduced precursors by etd (electron) can be found like
    # this:
    # mz*charge - (0, 1, 2, ..., prec_charge - 2)
    print('annotate spectra')
    for s in tqdm(spectra):
        prec = (s['mz'] * s['charge']) - s['charge'] + 1
        ind = match_float_to_array(prec, s['mz_ar'])
        if ind == -1:
            s['prec_int'] = 0
        else:
            s['prec_int'] = s['int_ar'][ind]
            
        if s['charge'] > 1:
            et_no_d = [(s['mz'] * s['charge']) - j for j in range(s['charge'] - 1)]
            s['et_no_d_int'] = sum([s['int_ar'][z] for z in [y for y in [match_float_to_array(x, s['mz_ar']) for x in et_no_d] if y != -1]])
        else:
            s['et_no_d_int'] = np.nan
    
    spectra = pd.DataFrame(spectra)
    
    # if the charge state of an precursor is unknown, MaxQuant copies
    # the spectra with different charge annotations.
    # Remove those spectra
    spectra = spectra[~spectra[['index', 'raw_file']].duplicated(keep=False)]
    # now the index is unique for each raw file
    spectra = spectra.set_index(['raw_file', 'index'])
    
    # load the identifications
    # For simplification, spectra that have second peptides are removed
    # Spectra with an empty matches column are also removed
    msms = pd.read_table(combined_dir + '/txt/msms.txt')
    msms = msms.drop_duplicates(['Raw file', 'Scan number'], keep=False)
    msms = msms[~msms['Matches'].isna()]
    msms.set_index(['Raw file', 'Scan number'], inplace=True)
    msms.index.names = ['raw_file', 'index']
    
    # join dfs
    spectra = spectra.join(msms, how='left')
    
    # add up fragment ion intensities
    spectra.loc[~spectra['Intensities'].isna(),'fragment_intensity'] = spectra[~spectra['Intensities'].isna()]['Intensities'].apply(lambda x: sum([float(y) for y in x.split(';')]))
    spectra['relative_fragmentation'] = spectra['fragment_intensity'] / spectra['prec_int']
    spectra['precursor_et_no_d'] = spectra['et_no_d_int'] / spectra['prec_int']
    # remove arrays
    spectra = spectra.drop(['mz_ar','int_ar'], axis=1)
    
    return(spectra)

spectra = main()

# save file
# the table is quite big and could be reduced to the important columns
spectra.to_csv(combined_dir + '/txt/spectra.txt', sep='\t')
