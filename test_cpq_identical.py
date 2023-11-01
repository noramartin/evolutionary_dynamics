#!/usr/bin/env python3
import NC_properties.neutral_component as NC
from NC_properties.RNA_seq_structure_functions import isundefinedpheno
import pandas as pd
import numpy as np
from collections import Counter
from os.path import isfile
from NC_properties.RNA_seq_structure_functions import get_mfe_structure_as_int

def c_pq_and_fraction_portals(GPmap, seq_vs_NCindex_array, NCindex_to_consider):
   K, L =  seq_vs_NCindex_array.shape[1], seq_vs_NCindex_array.ndim
   NC_size = np.sum(seq_vs_NCindex_array == NCindex_to_consider)
   p_vs_cpq = {}
   p_vs_fraction_portal_genotypes = {}
   for g1, NCindex in np.ndenumerate(seq_vs_NCindex_array):
      if NCindex_to_consider == NCindex:
         neighbour_phenos = Counter([GPmap[g2] for g2 in NC.neighbours_g(g1, K, L) if not isundefinedpheno(GPmap[g2])])
         for ph, c in neighbour_phenos.items():
            try:
               p_vs_cpq[ph] += 1.0* c/(L * (K-1) * NC_size)
               p_vs_fraction_portal_genotypes[ph] += 1.0/NC_size
            except KeyError:
               p_vs_cpq[ph] = 1.0* c/(L * (K-1) * NC_size)
               p_vs_fraction_portal_genotypes[ph] = 1.0/NC_size            	
         del neighbour_phenos
   print('finished function')
   return p_vs_cpq, p_vs_fraction_portal_genotypes


L = 12
df_NC = pd.read_csv('./RNA_data/RNA_NCs_L'+str(L)+'_forrandommap.csv')
NC_vs_size = {row['NCindex']: row['NC size'] for i, row in df_NC.iterrows()}

for NCindex_to_consider in NC_vs_size.keys():
   print('\n\n--------------------------------\n--------------------------------\nnew NC', NCindex_to_consider)
   filename_randommap = './randommap_data/randommapGPmap_L'+str(L)+'NCindex'+str(NCindex_to_consider)+'.npy'
   if not isfile(filename_randommap):
     continue

   ####################################################################################################
   print('input: RNA map', flush=True)
   ####################################################################################################
   filename_cpq = './RNA_data/RNA_NCs_cqp_data_L'+str(L)+'.csv'
   filename_rho = './RNA_data/RNA_NCs_L'+str(L)+'.csv'
   df_cpq, df_rho, NC_vs_p_vs_cpq, NC_vs_p_vs_number_portal_genotypes = pd.read_csv(filename_cpq), pd.read_csv(filename_rho), {}, {}
   for i, row in df_cpq.iterrows():
       try:
          NC_vs_p_vs_cpq[row['NCindex']][int(row['new structure integer'])] = row['cpq']
          NC_vs_p_vs_number_portal_genotypes[row['NCindex']][int(row['new structure integer'])] = row['number portal genotypes']
       except KeyError:
          NC_vs_p_vs_cpq[row['NCindex']] = {int(row['new structure integer']): row['cpq']}
          NC_vs_p_vs_number_portal_genotypes[row['NCindex']] = {int(row['new structure integer']): row['number portal genotypes']}
   NCindex_vs_rho = {row['NCindex']: row['rho'] for i, row in df_rho.iterrows()}

   ####################################################################################################
   print('test folding RNA data', flush=True)
   ####################################################################################################
   NCindex_arrayRNA = np.load('./RNA_data/RNA_NCindexArray_L'+str(L)+'.npy')
   GPmapRNA = np.load('./RNA_data/RNA_GPmap_L'+str(L)+'.npy')
   potential_genotypes = np.argwhere(NCindex_arrayRNA==NCindex_to_consider)
   assert potential_genotypes.shape[0] == NC_vs_size[NCindex_to_consider]
   phenotypes_set = set([])
   for i in range(NC_vs_size[NCindex_to_consider]):
      example_geno = tuple([x for x in potential_genotypes[i, :]])
      p0 = GPmapRNA[example_geno]
      phenotypes_set.add(p0)
      assert len(phenotypes_set) == 1
      assert p0 == get_mfe_structure_as_int(example_geno)
      for g2 in NC.neighbours_g(example_geno, 4, L):
         assert p0 != get_mfe_structure_as_int(g2) or NCindex_arrayRNA[g2] == NCindex_to_consider
      del example_geno
   example_geno = tuple([x for x in potential_genotypes[0, :]])
   ####################################################################################################
   print('test cpq for RNA data', flush=True)
   ####################################################################################################
   p_vs_cpq, p_vs_fraction_portal_genotypes = c_pq_and_fraction_portals(GPmapRNA, NCindex_arrayRNA, NCindex_to_consider)
   assert abs(p_vs_fraction_portal_genotypes[p0] -1 ) < 0.001
   assert abs(p_vs_cpq[p0]/NCindex_vs_rho[NCindex_to_consider] - 1) < 0.05 # rho correct
   for p1 in p_vs_cpq:
      if p_vs_cpq[p1] > 0:
         assert abs(p_vs_cpq[p1]/NC_vs_p_vs_cpq[NCindex_to_consider][p1] - 1) < 0.05 # cp1p0 correct
         assert abs(p_vs_fraction_portal_genotypes[p1]/NC_vs_p_vs_number_portal_genotypes[NCindex_to_consider][p1] - 1) < 0.05 # portals to p1 correct
         if abs(p_vs_cpq[p1]/NC_vs_p_vs_cpq[NCindex_to_consider][p1] - 1) > 0.001:
            print('deviation in phipq for RNA', NCindex_to_consider, NC_vs_size[NCindex_to_consider], p0, p1, p_vs_cpq[p1], NC_vs_p_vs_cpq[NCindex_to_consider][p1], flush=True)
         if abs(p_vs_fraction_portal_genotypes[p1]/NC_vs_p_vs_number_portal_genotypes[NCindex_to_consider][p1] - 1) > 0.001:
            print('deviation in phipq for RNA portals', NCindex_to_consider, NC_vs_size[NCindex_to_consider], p0, p1, p_vs_fraction_portal_genotypes[p1], NC_vs_p_vs_number_portal_genotypes[NCindex_to_consider][p1], flush=True)
   for set_other_NCs_in_neutral_set_to_zero in [True, False]:
      ####################################################################################################
      print('test against other data', flush=True)
      ####################################################################################################
      for type_map in ['randommap', 'topology_map', 'community_map', 'RNA']:
         print( type_map, set_other_NCs_in_neutral_set_to_zero)
         if (type_map == 'randommap' and set_other_NCs_in_neutral_set_to_zero) or (type_map == 'RNA' and not set_other_NCs_in_neutral_set_to_zero):
            continue
         elif type_map == 'randommap':
            GPmap = np.load('./'+type_map+'_data/'+type_map+'GPmap_L'+str(L)+'NCindex'+str(NCindex_to_consider)+'.npy')
            p_vs_cpq, p_vs_fraction_portal_genotypes = c_pq_and_fraction_portals(GPmap, GPmap, p0)
         elif set_other_NCs_in_neutral_set_to_zero:
            GPmap = np.load('./singleNC_data/'+type_map+'_GPmap_L'+str(L)+'NCindex'+str(NCindex_to_consider)+'_other_NCs_in_neutral_set_removed.npy')
            p_vs_cpq, p_vs_fraction_portal_genotypes = c_pq_and_fraction_portals(GPmap, NCindex_arrayRNA, NCindex_to_consider)
         else:
            GPmap = np.load('./'+type_map+'_data/'+type_map+'_GPmap_L'+str(L)+'NCindex'+str(NCindex_to_consider)+'.npy')
            p_vs_cpq, p_vs_fraction_portal_genotypes = c_pq_and_fraction_portals(GPmap, NCindex_arrayRNA, NCindex_to_consider)
         assert abs(p_vs_fraction_portal_genotypes[p0] -1 ) < 0.001
         assert abs(p_vs_cpq[p0]/NCindex_vs_rho[NCindex_to_consider] - 1) < 0.05 # rho correct
         if set_other_NCs_in_neutral_set_to_zero:
            type_map += '_otherNCszero'
         for p1 in p_vs_cpq:
            if p_vs_cpq[p1] > 0:
               #print(type_map, NCindex_to_consider, p0, p1, p_vs_cpq[p1], NC_vs_p_vs_cpq[NCindex_to_consider][p1], p_vs_cpq[p1]/NC_vs_p_vs_cpq[NCindex_to_consider][p1])
               if abs(p_vs_cpq[p1]/NC_vs_p_vs_cpq[NCindex_to_consider][p1] - 1) > 0.1: # cp1p0 correct
                  print('deviation in phipq', type_map, NCindex_to_consider, NC_vs_size[NCindex_to_consider], p0, p1, p_vs_cpq[p1], NC_vs_p_vs_cpq[NCindex_to_consider][p1], flush=True)
                  print('for this NC size, changing one mutational connection in the NC changes cpq by ', 1/(3 * L * NC_vs_size[NCindex_to_consider]), '\n')
                  assert type_map.startswith('topology_map') and p_vs_cpq[p1] < 0.01


      ####################################################################################################
      print('test that NC identical in other maps', flush=True)
      ####################################################################################################
      for type_map in ['topology_map', 'community_map']:
         print( type_map, set_other_NCs_in_neutral_set_to_zero)
         if set_other_NCs_in_neutral_set_to_zero:
            GPmap = np.load('./singleNC_data/'+type_map+'_GPmap_L'+str(L)+'NCindex'+str(NCindex_to_consider)+'_other_NCs_in_neutral_set_removed.npy')
         else:
            GPmap = np.load('./'+type_map+'_data/'+type_map+'_GPmap_L'+str(L)+'NCindex'+str(NCindex_to_consider)+'.npy')
         geno_vs_NC = NC.find_NC(p0, GPmap)
         NC_in_randomised_map = geno_vs_NC[example_geno]
         for g1, NCindex in np.ndenumerate(NCindex_arrayRNA):
             if NCindex == NCindex_to_consider:
             	  assert geno_vs_NC[g1] == NC_in_randomised_map
         assert len([NCindex for g, NCindex in geno_vs_NC.items() if NCindex == NC_in_randomised_map ]) == NC_vs_size[NCindex_to_consider]




