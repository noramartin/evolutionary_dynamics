#!/usr/bin/env python3
import sys
sys.path.insert(0,'/mnt/zfsusers/nmartin/ViennaRNA/lib/python3.6/site-packages')
import numpy as np
import pandas as pd
from os.path import isfile
import networkx as nx
from copy import deepcopy
from collections import Counter

def NC_vs_cqp_info(GPmap, seq_vs_NCindex_array, ph_list):
   NC_indices_withundefined, NC_sizes_withundefined = np.unique(seq_vs_NCindex_array, return_counts=True)
   NCindex_vs_size = {NCindex: size for NCindex, size in zip(NC_indices_withundefined, NC_sizes_withundefined) if NCindex > 0}
   NC_vs_p_vs_cpq = {NCindex: {} for NCindex in NCindex_vs_size if NCindex_vs_size[NCindex] > 2}
   NC_vs_p_vs_fraction_portal_genotypes = {NCindex: {} for NCindex in NCindex_vs_size if NCindex_vs_size[NCindex] > 2}
   max_index_by_site = [s - 1 for s in GPmap.shape]
   for g1, NCindex in np.ndenumerate(seq_vs_NCindex_array):
      if NCindex > 0 and NCindex_vs_size[NCindex] > 2:
         assert NCindex == seq_vs_NCindex_array[g1]
         neighbours = [GPmap[g2] for g2 in neighbours_g_indices_nonperiodic_boundary(g1, max_index_by_site)]
         neighbour_phenos = Counter(neighbours)
         for ph, c in neighbour_phenos.items():
            try:
               NC_vs_p_vs_cpq[NCindex][ph] += 1.0* c/(len(neighbours) * NCindex_vs_size[NCindex])
               NC_vs_p_vs_fraction_portal_genotypes[NCindex][ph] += 1.0/NCindex_vs_size[NCindex]
            except KeyError:
               NC_vs_p_vs_cpq[NCindex][ph] = 1.0* c/(len(neighbours) * NCindex_vs_size[NCindex])
               NC_vs_p_vs_fraction_portal_genotypes[NCindex][ph] = 1.0/NCindex_vs_size[NCindex]
         del neighbour_phenos
   print('finished function')
   return NC_vs_p_vs_cpq, NC_vs_p_vs_fraction_portal_genotypes

def neighbours_g_indices_nonperiodic_boundary(g, max_index_by_site):  
   return [tuple(n[:]) for site in range(len(g)) for n in neighbours_g_indices_nonperiodic_boundary_given_site(g, max_index_by_site, site) ] 

def neighbours_g_indices_nonperiodic_boundary_given_site(g, max_index_by_site, site):  
   new_indices = [g[site] -1, g[site] +1]
   return [[g[site_index] if site_index != site else new_site for site_index in range(len(g))] for new_site in  new_indices if new_site >= 0 and new_site <= max_index_by_site[site]]

###############################################################################################
###############################################################################################
str_GPmapparams = '3_1_8_30_5'
#genemax, g9min, g9max = tuple([str_GPmapparams.split('_')[i] for i in range(3)])
GPmap = np.load('./biomorph_data/phenotypes'+str_GPmapparams+'.npy')
#################################################
print("find neutral components", flush=True)
#################################################
NCindex_array = np.load('./biomorph_data/neutral_components'+str_GPmapparams+'.npy')
if not isfile('./biomorph_data/biomorph_NCs_'+str_GPmapparams+'.csv'):
   NC_indices_withundefined, NC_sizes_withundefined = np.unique(NCindex_array, return_counts=True)
   NC_indices = [NCindex for NCindex in NC_indices_withundefined if NCindex > 0]
   NC_sizes = [NC_sizes_withundefined[i] for i, NCindex in enumerate(NC_indices_withundefined) if NCindex > 0]
   NCindex_vs_ph = {NCindex: GPmap[g1] for g1, NCindex in np.ndenumerate(NCindex_array)}
   structures_NC_indices = [NCindex_vs_ph[NCindex] for NCindex in NC_indices]
   df_NC = pd.DataFrame.from_dict({'NCindex': list(NC_indices),
                            'structure integer': structures_NC_indices,
                            'NC size': list(NC_sizes)})
   df_NC.to_csv('./biomorph_data/biomorph_NCs_'+str_GPmapparams+'.csv')
   del structures_NC_indices
else:
  df_NC = pd.read_csv('./biomorph_data/biomorph_NCs_'+str_GPmapparams+'.csv')
  NC_indices = df_NC['NCindex'].tolist()
NC_vs_size = {NCindex: NCsize for NCindex, NCsize in zip(NC_indices, df_NC['NC size'].tolist())}
NC_vs_structure = {NCindex: NCsize for NCindex, NCsize in zip(NC_indices, df_NC['structure integer'].tolist())}
assert NCindex_array.dtype in [np.uint32, np.int]
#################################################
print("find connections between NCs", flush=True)
#################################################
filename_cpq = './biomorph_data/biomorph_NCs_cqp_data'+str_GPmapparams+'.csv'
ph_list = [ph for ph in np.unique(GPmap)]
if not isfile(filename_cpq):  
   NC_vs_p_vs_cpq, NC_vs_p_vs_fraction_portal_genotypes = NC_vs_cqp_info(GPmap, NCindex_array, ph_list)
   NC_indices = [NCindex for NCindex in NC_vs_p_vs_cpq]
   NCindex_list_for_df = [NCindex for NCindex in list(NC_indices) for p in NC_vs_p_vs_cpq[NCindex]]
   new_structure_list_for_df = [p for NCindex in list(NC_indices) for p in NC_vs_p_vs_cpq[NCindex]]
   cpq_list_for_df = [NC_vs_p_vs_cpq[NCindex][p] for NCindex in list(NC_indices) for p in NC_vs_p_vs_cpq[NCindex]]
   portal_list_df = [NC_vs_p_vs_fraction_portal_genotypes[NCindex][p] for NCindex in list(NC_indices) for p in NC_vs_p_vs_cpq[NCindex]]
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
NCindices_to_use = list(np.random.choice(NCindices_to_choose_from, 5))  + list(np.random.choice(NCindices_to_choose_from, 5))#+ [257, ]
for NCindex in NCindices_to_use:
   if NCindex not in NCindices_to_use:
     df_NC.drop(df_NC[df_NC.NCindex == NCindex].index, inplace=True)
df_NC = pd.DataFrame.from_dict({'NCindex': NCindices_to_use,
                            'structure integer': [NC_vs_structure[NCindex] for NCindex in NCindices_to_use],
                            'NC size': [NC_vs_size[NCindex] for NCindex in NCindices_to_use]})
df_NC.to_csv('./biomorph_data/biomorph_NCs_'+str_GPmapparams+'_forsimulation.csv')



