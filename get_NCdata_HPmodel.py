#!/usr/bin/env python3
import sys
sys.path.insert(0,'/mnt/zfsusers/nmartin/ViennaRNA/lib/python3.6/site-packages')
import numpy as np
import NC_properties.neutral_component as NC
import pandas as pd
from os.path import isfile
import networkx as nx
from copy import deepcopy
import parameters_HP as param

def is_unfolded_HP_structure(s):
   return s < 0.5

###############################################################################################
###############################################################################################
GPmap = np.load(param.GPmap_filename)
K, L = int(GPmap.shape[1]), int(GPmap.ndim)
#################################################
print("find neutral components", flush=True)
#################################################
NCindex_array = NC.find_all_NCs_parallel(GPmap, filename='./HP_data/HP_NCindexArray_L'+str(L)+'.npy', isundefinedphenofunction=is_unfolded_HP_structure)
if not isfile('./HP_data/HP_NCs_L'+str(L)+'.csv'):
   NC_indices_withundefined, NC_sizes_withundefined = np.unique(NCindex_array, return_counts=True)
   NC_indices = [NCindex for NCindex in NC_indices_withundefined if NCindex > 0]
   NC_sizes = [NC_sizes_withundefined[i] for i, NCindex in enumerate(NC_indices_withundefined) if NCindex > 0]
   structures_NC_indices = [GPmap[NC.find_index(NCindex, NCindex_array)] for NCindex in NC_indices]
   NC_vs_rho = NC.NC_vs_rho(GPmap, NCindex_array)
   df_NC = pd.DataFrame.from_dict({'NCindex': list(NC_indices),
                            'structure integer': structures_NC_indices,
                            'example sequence': [''.join([str(s) for s in NC.find_index(NCindex, NCindex_array)]) for NCindex in NC_indices],
                            'structure': [param.contact_map_vs_updown[param.contact_map_list[s-1]] for s in structures_NC_indices],
                            'NC size': list(NC_sizes),
                            'rho': [NC_vs_rho[NCindex] for NCindex in list(NC_indices)]})
   df_NC.to_csv('./HP_data/HP_NCs_L'+str(L)+'.csv')
   del structures_NC_indices
else:
  df_NC = pd.read_csv('./HP_data/HP_NCs_L'+str(L)+'.csv')
  NC_indices = df_NC['NCindex'].tolist()
NC_vs_size = {NCindex: NCsize for NCindex, NCsize in zip(NC_indices, df_NC['NC size'].tolist())}
assert NCindex_array.dtype in [np.uint32, np.int]
#################################################
print("find connections between NCs", flush=True)
#################################################
filename_cpq = './HP_data/HP_NCs_cqp_data_L'+str(L)+'.csv'
ph_list = [ph for ph in np.unique(GPmap)] # if not is_unfolded_HP_structure(ph)]
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
print("which NCs to choose for simulations", flush=True)
##################################################################################################
##################################################################################################
#NCindices_for_drawing = [312, 310, 241]
np.random.seed(3)
NCindices_to_choose_from = [NCindex for NCindex, NCsize in NC_vs_size.items() if NCsize > 10**3]
NCindices_to_use = list(np.random.choice(NCindices_to_choose_from, 5)) #+ [257, ]
for NCindex in NC_vs_size.keys():
   if NCindex not in NCindices_to_use:
     df_NC.drop(df_NC[df_NC.NCindex == NCindex].index, inplace=True)
df_NC.to_csv('./HP_data/HP_NCs_L'+str(L)+'_forsimulation.csv')



