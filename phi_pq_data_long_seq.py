#!/usr/bin/env python
import numpy as np
import pandas as pd
from collections import Counter
from NC_properties.RNA_seq_structure_functions import get_mfe_structure, dotbracket_to_coarsegrained, sequence_int_to_str, sequence_str_to_int
from NC_properties.neutral_component import neighbours_g
from NC_properties.own_implementation_Weiss_Ahnert_community_detection import neighbours_g_given_site
from copy import deepcopy
from os.path import isfile
from multiprocessing import Pool
from functools import partial


def g_sampling_structure_sample(required_sample_size, L, folding_function):
   """adapted from free-energy-based neutral set size prediction code"""
   structure_vs_seq = {}
   while len(structure_vs_seq) < required_sample_size * 10**3:
      sequence_sample_list = [tuple(np.random.choice([0, 1, 2, 3], replace=True, size=L)) for i in range(10**3)]     
      p = Pool(10)
      with p:
         pool_result = p.map(folding_function, sequence_sample_list)
      for seq, struct in zip(sequence_sample_list, pool_result):
          if struct not in structure_vs_seq:
             structure_vs_seq[deepcopy(struct)] = deepcopy(seq)
   structure_list = list([s for s in structure_vs_seq.keys() if s.count('(') > 0])
   no_stacks_list = [dotbracket_to_coarsegrained(s).count('[') for s in structure_list]
   no_stacks_vs_freq = Counter(no_stacks_list)
   if len(no_stacks_vs_freq) < required_sample_size:
      weight_list = [1/float(no_stacks_vs_freq[n]) for n in no_stacks_list]
      weight_list_normalised = np.divide(weight_list, float(np.sum(weight_list)))
      final_structure_list = list(np.random.choice(structure_list, required_sample_size, p=weight_list_normalised, replace=False))
   else:
      no_stacks_chosen = np.random.choice([n for n in no_stacks_vs_freq.keys()], replace=False, size=required_sample_size)
      final_structure_list = [np.random.choice([structure for structure, n_stacks in zip(structure_list, no_stacks_list) if n_stacks == n]) for n in no_stacks_chosen]
   return final_structure_list, [structure_vs_seq[structure] for structure in final_structure_list]

def rw_sitescanning_nobpswaps(startseq, length_per_walk, every_Nth_seq, folding_function):
   """perform a site-scanning random walk of length length_per_walk starting from startseq (tuple of ints) and subsample every_Nth_seq;
   site-scanning method following Weiss and Ahnert (2020), Royal Society Interface"""
   seq_list_rw, current_seq, dotbracket_structure = [], tuple([g for g in startseq]), folding_function(startseq)
   L, K, site_to_scan = len(startseq), 4, 0
   seq_list_rw.append(current_seq)
   neutral_neighbours_startseq = [g_mut for g_mut in neighbours_g(current_seq, K, L) if dotbracket_structure == folding_function(g_mut)]
   if len(neutral_neighbours_startseq) == 0:
      return [current_seq,]
   while len(seq_list_rw) < length_per_walk:
      neighbours_given_site = neighbours_g_given_site(current_seq, K, L, site_to_scan)
      neighbours_given_site_shuffled = [tuple(neighbours_given_site[i]) for i in np.random.choice(len(neighbours_given_site), size=len(neighbours_given_site), replace=False)]
      for g in neighbours_given_site_shuffled:
         if dotbracket_structure == folding_function(g):
            current_seq = tuple([c for c in g])
            seq_list_rw.append(deepcopy(current_seq))
            break
      site_to_scan = (site_to_scan+1)%L  
   assert len(seq_list_rw) == length_per_walk
   return [tuple(deepcopy(seq_list_rw[i])) for i in np.random.choice(length_per_walk, size=length_per_walk//every_Nth_seq, replace=False)]
      

def get_phipq_sequence_sample(sequence_list, folding_function):
   print('start phipq calculation')
   phi_pq = {}
   L = len(sequence_list[0])
   K = 4
   for seq in sequence_list:
      neighbours = neighbours_g(tuple(seq), K, L)
      for neighbourgeno in neighbours:
         neighbourpheno = folding_function(neighbourgeno)
         try:
            phi_pq[neighbourpheno] += 1.0/(len(sequence_list) * 3 * L)
         except KeyError:
            phi_pq[neighbourpheno] = 1.0/(len(sequence_list) * 3 * L)  
   return phi_pq

###############################################################################################
###############################################################################################
print(  'parameters')
###############################################################################################
###############################################################################################
L = 30
required_sample_size, length_per_walk, every_Nth_seq = 5, 10**5, 20
###############
filename_structure_list = './data_long_seq/structure_list_L' + str(L) + '_sample' + str(required_sample_size) + '.csv'
filename_seq_sample = './data_long_seq/sequence_sample_L' + str(L) + '_sample' + str(required_sample_size) +'_rwparams' + str(length_per_walk) +'_' + str(every_Nth_seq) + '.csv'
phipq_filename_sampling = './data_long_seq/phi_pq_data' + str(L) + '_sample' + str(required_sample_size) +'_rwparams' + str(length_per_walk) +'_'  + str(every_Nth_seq) + '.csv'
###############################################################################################
###############################################################################################
print(  'select a few structures with different number of stacks')
###############################################################################################
###############################################################################################
if not isfile(filename_structure_list):
   structure_sample, start_seq_list = g_sampling_structure_sample(required_sample_size, L, folding_function=get_mfe_structure)
   df_structures = pd.DataFrame.from_dict({'structure': structure_sample, 'start sequence': [sequence_int_to_str(seq) for seq in start_seq_list]})
   df_structures.to_csv(filename_structure_list)
else:
   df_structures = pd.read_csv(filename_structure_list)
   structure_sample = df_structures['structure'].tolist()
   start_seq_list = [sequence_str_to_int(seq[:]) for seq in df_structures['start sequence'].tolist()] 
   for struct, seq in zip(structure_sample, start_seq_list):
      assert get_mfe_structure(seq) == struct
###############################################################################################
###############################################################################################
print(  'sequence sample for phipq data')
###############################################################################################
###############################################################################################
if not isfile(filename_seq_sample):
   site_scanning_function = partial(rw_sitescanning_nobpswaps, 
                                    length_per_walk=length_per_walk, every_Nth_seq=every_Nth_seq, folding_function=get_mfe_structure)
   p = Pool(10)
   with p:
      seq_list_each_structure_list = p.map(site_scanning_function, start_seq_list)
   for i, structure in enumerate(structure_sample):
      if len(seq_list_each_structure_list[i]) == 0:
         df_structures.drop(df_structures[df_structures.structure == structure].index, inplace=True)
   df_structures.to_csv(filename_structure_list)
   structure_sample_df = pd.DataFrame.from_dict({'structure': [structure_sample[i][:] for i, seq_list in enumerate(seq_list_each_structure_list) for seq in seq_list],
                                                       'seq': [sequence_int_to_str(seq) for i, seq_list in enumerate(seq_list_each_structure_list) for seq in seq_list]})
   structure_sample_df.to_csv(filename_seq_sample)
else:
   structure_sample_df = pd.read_csv(filename_seq_sample)
   struct_vs_seq = {structure: [] for structure in structure_sample}
   for i, row in structure_sample_df.iterrows(): 
      struct_vs_seq[row['structure']].append(row['seq'])
   seq_list_each_structure_list = [[sequence_str_to_int(seq[:]) for seq in struct_vs_seq[s]] for s in structure_sample]
   print(sorted([s for s in struct_vs_seq.keys()]))
   print(sorted(df_structures['structure'].tolist()))
###############################################################################################
###############################################################################################
print('get phipq stats for each sequence sample')
###############################################################################################
###############################################################################################
if not isfile(phipq_filename_sampling):
   function_for_parallelised_calc = partial(get_phipq_sequence_sample, 
                                                     folding_function=get_mfe_structure)
            
   p = Pool(10)
   with p:
      pool_result = p.map(function_for_parallelised_calc, seq_list_each_structure_list)
   phi_vs_phi_pq_allstructures = {structure: deepcopy(pool_result[structureindex]) for structureindex, structure in enumerate(structure_sample)} 
   structure_one_list, structure_two_list, phi_pq_list = zip(*[(structureone, structuretwo, phi) for structureone in structure_sample for structuretwo, phi in phi_vs_phi_pq_allstructures[structureone].items()])
   df_phipq = pd.DataFrame.from_dict({'structure 1': structure_one_list, 'structure 2': structure_two_list, 'phi': phi_pq_list})
   df_phipq.to_csv(phipq_filename_sampling)
       

