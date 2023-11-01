#!/usr/bin/env python3
import sys
sys.path.insert(0,'/mnt/zfsusers/nmartin/ViennaRNA/lib/python3.6/site-packages')
import numpy as np
from multiprocessing import Pool
from NC_properties.RNA_seq_structure_functions import *
import NC_properties.neutral_component as NC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import NC_properties.randommap_construction as randommap
import pandas as pd
from os import cpu_count
from os.path import isfile
from NC_properties.own_implementation_Weiss_Ahnert_community_detection import find_communities
import networkx as nx
from scipy.stats import pearsonr
import seaborn as sns
from copy import deepcopy

L = 12 #12
K = 4

##################################################################################################
##################################################################################################
print("RNA GP map: save data")
##################################################################################################
##################################################################################################
#################################################
print("build GP map")
#################################################
GPmapfilename = './RNA_data/RNA_GPmap_L'+str(L)+'.npy'
if not isfile(GPmapfilename):
   GPmap = np.zeros((K,)*L, dtype='uint32')
   #genotype_list = [g for g, ph in np.ndenumerate(GPmap)]
   #with Pool(10) as pool:
   #   ph_list = pool.map(get_mfe_structure_as_int, genotype_list)
   progress = 0
   #for g_index, genotype in enumerate(genotype_list):
   for g, ph in np.ndenumerate(GPmap):
      mfe_struct = get_mfe_structure_as_int(g)
      assert mfe_struct < 2**32 #limit of array
      GPmap[g] = mfe_struct #ph_list[g_index]
      progress +=1
      if progress % 10**4 ==0:
         print('finished folding', progress/K**L * 100, '% of all sequences')
   np.save(GPmapfilename, GPmap, allow_pickle=False)
   #del genotype_list, ph_list
else:
   GPmap = np.load(GPmapfilename)
print(GPmap.dtype, GPmap.dtype.kind)
assert GPmap.dtype in [np.uint32, np.int]

#################################################
print("find neutral set sizes", flush=True)
#################################################
Ndf_filename = './RNA_data/RNA_neutral_set_size_L'+str(L)+'.csv'
if not isfile(Ndf_filename):
   structure_integers, structure_counts = np.unique(GPmap, return_counts=True)
   df_neutral_sets = pd.DataFrame.from_dict({'structure integer': list(structure_integers),
                            'structure': [get_dotbracket_from_int(s) for s in structure_integers],
                            'neutral set size': list(structure_counts)})
   df_neutral_sets.to_csv(Ndf_filename)
else:
  df_neutral_sets = pd.read_csv(Ndf_filename)
#################################################
print("find phi_pq and rho")
#################################################
filename_phipq = './RNA_data/RNA_phiqp_and_rho_data_L'+str(L)+'.csv'
ph_list = df_neutral_sets['structure integer'].tolist()
ph_vs_size = {p: N for p, N in zip(df_neutral_sets['structure integer'].tolist(), df_neutral_sets['neutral set size'].tolist())}
if not isfile(filename_phipq):  
   q_vs_p_vs_phipq_and_rho = NC.phi_pq_GPmap(GPmap, ph_vs_size)
   old_structure_list_for_df = [q for q in ph_list for p in ph_list]
   new_structure_list_for_df = [p for q in ph_list for p in ph_list]
   phipq_list_for_df = [q_vs_p_vs_phipq_and_rho[q][p] for q in ph_list for p in ph_list]
   df_cpq = pd.DataFrame.from_dict({'old structure integer': old_structure_list_for_df,
                               'new structure integer': new_structure_list_for_df,
                               'phipq': phipq_list_for_df})
   df_cpq.to_csv(filename_phipq)
##################################################################################################

#################################################
print("find neutral components", flush=True)
#################################################
GPmap = np.load(GPmapfilename)
NCindex_array = NC.find_all_NCs_parallel(GPmap, filename='./RNA_data/RNA_NCindexArray_L'+str(L)+'.npy')
if not isfile('./RNA_data/RNA_NCs_L'+str(L)+'.csv'):
   NC_indices_withundefined, NC_sizes_withundefined = np.unique(NCindex_array, return_counts=True)
   NC_indices = [NCindex for NCindex in NC_indices_withundefined if NCindex > 0]
   NC_sizes = [NC_sizes_withundefined[i] for i, NCindex in enumerate(NC_indices_withundefined) if NCindex > 0]
   structures_NC_indices = [GPmap[NC.find_index(NCindex, NCindex_array)] for NCindex in NC_indices]
   NC_vs_rho = NC.NC_vs_rho(GPmap, NCindex_array)
   dfNC = pd.DataFrame.from_dict({'NCindex': list(NC_indices),
                            'structure integer': structures_NC_indices,
                            'example sequence': [sequence_int_to_str(NC.find_index(NCindex, NCindex_array)) for NCindex in NC_indices],
                            'structure': [get_dotbracket_from_int(s) for s in structures_NC_indices],
                            'NC size': list(NC_sizes),
                            'rho': [NC_vs_rho[NCindex] for NCindex in list(NC_indices)]})
   dfNC.to_csv('./RNA_data/RNA_NCs_L'+str(L)+'.csv')
   del structures_NC_indices
else:
  dfNC = pd.read_csv('./RNA_data/RNA_NCs_L'+str(L)+'.csv')
  NC_indices = dfNC['NCindex'].tolist()
NC_vs_size = {NCindex: NCsize for NCindex, NCsize in zip(NC_indices, dfNC['NC size'].tolist())}
NC_vs_structure = {NCindex: NCsize for NCindex, NCsize in zip(NC_indices, dfNC['structure integer'].tolist())}
NC_vs_rho = {NCindex: NCsize for NCindex, NCsize in zip(NC_indices, dfNC['rho'].tolist())}
del dfNC
NCindices_by_size = sorted(NC_indices, key=NC_vs_size.get, reverse=True)
assert NCindex_array.dtype in [np.uint32, np.int]
#################################################
print("find connections between NCs", flush=True)
#################################################
filename_cpq = './RNA_data/RNA_NCs_cqp_data_L'+str(L)+'.csv'
ph_list = df_neutral_sets['structure integer'].tolist()
if not isfile(filename_cpq):  
   NC_vs_p_vs_cpq, NC_vs_p_vs_fraction_portal_genotypes = NC.NC_vs_cqp_info(GPmap, NCindex_array, ph_list)
   NC_indices = [NCindex for NCindex in NC_vs_p_vs_cpq]
   NCindex_list_for_df = [NCindex for NCindex in list(NC_indices) for p in ph_list]
   new_structure_list_for_df = [p for NCindex in list(NC_indices) for p in ph_list]
   cpq_list_for_df = [NC_vs_p_vs_cpq[NCindex][p] for NCindex in list(NC_indices) for p in ph_list]
   portal_list_df = [NC_vs_p_vs_fraction_portal_genotypes[NCindex][p] for NCindex in list(NC_indices) for p in ph_list]
   df_cpq = pd.DataFrame.from_dict({'NCindex': NCindex_list_for_df,
                               'new structure integer': new_structure_list_for_df,
                               'cpq': cpq_list_for_df,
                               'number portal genotypes': portal_list_df})
   df_cpq.to_csv(filename_cpq)
else:
   print('load data')
   df_cpq = pd.read_csv(filename_cpq)
   #NC_indices = list(set([NCindex for NCindex in df_cpq['NCindex'].tolist()]))
   NC_vs_p_vs_cpq = {NCindex: {} for NCindex in NC_indices}
   NC_vs_p_vs_fraction_portal_genotypes = {NCindex: {} for NCindex in NC_indices}
   for i, row in df_cpq.iterrows():
      NC_vs_p_vs_cpq[int(row['NCindex'])][int(row['new structure integer'])] = row['cpq']
      NC_vs_p_vs_fraction_portal_genotypes[int(row['NCindex'])][int(row['new structure integer'])] = row['number portal genotypes']
##################################################################################################
##################################################################################################



##################################################################################################
##################################################################################################
print("which NCs to choose for randommap", flush=True)
##################################################################################################
##################################################################################################
#NCindices_for_drawing = [312, 310, 241]
np.random.seed(3)
GPmapRNA = np.load('./RNA_data/RNA_GPmap_L'+str(L)+'.npy') #gp map
df_NC = pd.read_csv('./RNA_data/RNA_NCs_L'+str(L)+'.csv')
NCindex_array = NC.find_all_NCs_parallel(GPmapRNA, filename='./RNA_data/RNA_NCindexArray_L'+str(L)+'.npy')
NC_vs_size = {row['NCindex']: row['NC size'] for i, row in df_NC.iterrows()}
NCindices_to_choose_from = [NCindex for NCindex, NCsize in NC_vs_size.items() if NCsize > 10**3]
NCindices_to_use = list(np.random.choice(NCindices_to_choose_from, 5)) #+ [257, ]
del df_NC
#### also for two-peaked
#if isfile('./RNA_data/p0_p1_p2_candidates.csv'):
#   df_twopeaked = pd.read_csv('./RNA_data/p0_p1_p2_candidates.csv')
#   NCindices_for_twopeaked = list(set(df_twopeaked['NCindex'].tolist()))
#else:
NCindices_for_twopeaked = [NCindex_array[sequence_str_to_int('ACCUAAAAAAGG')]]
##################################################################################################
##################################################################################################
print("find communitites in each NC", flush=True)
##################################################################################################
##################################################################################################
for NCindex in NCindices_to_use + NCindices_for_twopeaked:
  print('communitites for', NCindex)
  structure_int = NC_vs_structure[NCindex]
  filename_partition = './RNA_data/RNA_NC'+str(NCindex)+'partition_L'+str(L)+'.csv'
  if not isfile(filename_partition):
     best_partition = find_communities(NCindex, NCindex_array)
     ### save
     partition_index_list, genotype_list = zip(*[(index, sequence_int_to_str(g)) for index, l in enumerate(best_partition) for g in l])
     df_partition = pd.DataFrame.from_dict({'partition index': partition_index_list, 'genotype': genotype_list})
     df_partition.to_csv(filename_partition)

##################################################################################################
##################################################################################################
print("randommap GP map: save data", flush=True)
##################################################################################################
##################################################################################################
for NCindex in NCindices_to_use + NCindices_for_twopeaked:
   filename_shuffled = './randommap_data/randommapGPmap_L'+str(L)+'NCindex'+str(NCindex)+'.npy'
   if not isfile(filename_shuffled):
      GPmap_randommap = randommap.create_randommap(GPmapRNA, NCindex_array, NCindex, phi_pq_local=NC_vs_p_vs_cpq[NCindex])
      assert GPmap_randommap.dtype in [np.uint32, np.int]
      np.save(filename_shuffled, GPmap_randommap, allow_pickle=False)
      del GPmap_randommap

##################################################################################################
##################################################################################################
print("community-preserving randommap GP map: save data", flush=True)
##################################################################################################
##################################################################################################
for NCindex in NCindices_to_use + NCindices_for_twopeaked:
   filename_shuffled = './topology_map_data/topology_map_GPmap_L'+str(L)+'NCindex'+str(NCindex)+'.npy'
   if not isfile(filename_shuffled):
      GPmap_shuffled = randommap.create_topology_preserving_randommap_second_method(GPmapRNA, NCindex_array, NCindex, phi_pq_local=NC_vs_p_vs_cpq[NCindex])
      assert GPmap_shuffled.dtype in [np.uint32, np.int]
      np.save(filename_shuffled, GPmap_shuffled, allow_pickle=False)
##################################################################################################
##################################################################################################
print("community-preserving randommap which preserves neighbourhood around communitites: save data")
##################################################################################################
##################################################################################################
for NCindex in NCindices_to_use + NCindices_for_twopeaked:
  filename_shuffled ='./community_map_data/community_map_GPmap_L'+str(L)+'NCindex'+str(NCindex)+'.npy'
  if not isfile(filename_shuffled):
     df_partition = pd.read_csv('./RNA_data/RNA_NC'+str(NCindex)+'partition_L'+str(L)+'.csv' )
     genotype_vs_partition = {sequence_str_to_int(row['genotype']): row['partition index'] for i, row in df_partition.iterrows()}
     GPmap_shuffled = randommap.create_topology_preserving_randommap_with_community_neighbourhood_intact(GPmapRNA, genotype_vs_partition)
     assert GPmap_shuffled.dtype in [np.uint32, np.int]
     np.save(filename_shuffled, GPmap_shuffled, allow_pickle=False)

#################################################
print("summary statistics only if randommap saved")
#################################################
dfNC = pd.read_csv('./RNA_data/RNA_NCs_L'+str(L)+'.csv')
for NCindex in NC_vs_size.keys():
   if NCindex not in NCindices_to_use:
     dfNC.drop(dfNC[dfNC.NCindex == NCindex].index, inplace=True)
dfNC.to_csv('./RNA_data/RNA_NCs_L'+str(L)+'_forrandommap.csv')

#################################################
print("check that different maps differ")
#################################################
for NCindex in NC_vs_size.keys():
  filename_randomRNAncNeighbourhood ='./community_map_data/community_map_GPmap_L'+str(L)+'NCindex'+str(NCindex)+'.npy'
  filename_randomRNAnc = './topology_map_data/topology_map_GPmap_L'+str(L)+'NCindex'+str(NCindex)+'.npy'
  filename_randommap = './randommap_data/randommapGPmap_L'+str(L)+'NCindex'+str(NCindex)+'.npy'
  if not isfile(filename_randommap):
    continue
  for filename in [filename_randomRNAncNeighbourhood, filename_randomRNAnc, filename_randommap]:
     GPmap_nullmodel = np.load(filename)
     assert GPmap_nullmodel.dtype in [np.uint32, np.int]
     assert np.sum(np.abs(GPmap - GPmap_nullmodel)) > 1
     print(filename, '\n', np.sum(GPmap != GPmap_nullmodel), 'NC size', NC_vs_size[NCindex], '\n\n\n')

##################################################################################################
##################################################################################################
print("for all selected NCs - save a GP map that sets other part of neutral set to undefined")
##################################################################################################
##################################################################################################
for NCindex in NCindices_to_use + NCindices_for_twopeaked:
   phenotype = GPmapRNA[NC.find_index(NCindex, NCindex_array)]
   for type_map in ['topology_map', 'community_map', 'RNA']:
      if not isfile('./singleNC_data/'+type_map+'_GPmap_L'+str(L)+'NCindex'+str(NCindex)+'_other_NCs_in_neutral_set_removed.npy'):
         if type_map == 'RNA':
            GPmap_to_change = np.load('./'+type_map+'_data/'+type_map+'_GPmap_L'+str(L)+'.npy')
         else:
            GPmap_to_change = np.load('./'+type_map+'_data/'+type_map+'_GPmap_L'+str(L)+'NCindex'+str(NCindex)+'.npy') 
         GPmap_single_NC = deepcopy(GPmap_to_change.copy())
         for g, ph in np.ndenumerate(GPmap_single_NC):
           if ph == phenotype and NCindex_array[tuple(g)] != NCindex:
             GPmap_single_NC[tuple(g)] = -1
         assert GPmap_single_NC.dtype in [np.uint32, np.int]
         np.save('./singleNC_data/'+type_map+'_GPmap_L'+str(L)+'NCindex'+str(NCindex)+'_other_NCs_in_neutral_set_removed.npy', GPmap_single_NC)




