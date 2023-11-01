#!/usr/bin/env python3
import numpy as np
import sys
print('python version number', sys.version_info, flush=True)
from evolution_functions.evolutionary_dynamics_function_numba import *
from evolution_functions.evolutionary_dynamics_function_biomorphs import run_WrightFisher_biomorphs
import NC_properties.neutral_component as NC
import pandas as pd
import NC_properties.general_functions as g
from os.path import isfile


################################################################################################################################################################################

type_map=sys.argv[1]
assert type_map in ['RNA', 'randommap', 'topology_map', 'community_map', 'meanfield', 'HPmodel', 'biomorphs']
################################################################################################################################################################################
print('population parameters', flush=True)
################################################################################################################################################################################
N, mu = int(sys.argv[2]), float(sys.argv[3])  #5 * 10**7
set_other_NCs_in_neutral_set_to_zero = int(sys.argv[4])
adjust_T = int(sys.argv[5])
################################################################################################################################################################################
rare_pheno_min_f, rare_pheno_max_f = 0.001, 0.1
L = 12
################################################################################################################################################################################
if not adjust_T:
   T = 10**7
else:
   T = max(g.roundT_up(10**3/(N * mu * rare_pheno_min_f * L)), 10**4) # at least 1000 found of the rarest type  (on average)
   print('N, mu, T', N, mu, T)
################################################################################################################################################################################
string_saving = 'neutral_N'+g.format_integer_filename(N)+'_T'+g.format_integer_filename(T)+'_mu'+g.format_float_filename(mu)+'minf'+g.format_float_filename(rare_pheno_min_f)+'maxf'+g.format_float_filename(rare_pheno_max_f)
if set_other_NCs_in_neutral_set_to_zero:
   string_saving += 'otherNCszero'
   print(string_saving)
print('N, mu, T', N, mu, T)
if type_map != 'HPmodel' and type_map != 'biomorphs':
   ####################################################################################################
   print('load GP map properties', flush=True)
   ####################################################################################################
   df = pd.read_csv('./RNA_data/RNA_NCs_L'+str(L)+'_forrandommap.csv')
   NCindex_list = df['NCindex'].tolist()
   NCindex_vs_structure = {NCindex: structure_int for NCindex, structure_int in zip(df['NCindex'].tolist(), df['structure integer'].tolist())}
   GPmapRNA=np.load('./RNA_data/RNA_GPmap_L'+str(L)+'.npy')
   NCindex_arrayRNA = NC.find_all_NCs_parallel(GPmapRNA, filename='./RNA_data/RNA_NCindexArray_L'+str(L)+'.npy')
   ####################################################################################################
   print('load cpq values', flush=True)
   ####################################################################################################
   filename_cpq = './RNA_data/RNA_NCs_cqp_data_L'+str(L)+'.csv'
   df_cpq, NC_vs_p_vs_cpq = pd.read_csv(filename_cpq), {}
   for i, row in df_cpq.iterrows():
       try:
          NC_vs_p_vs_cpq[row['NCindex']][int(row['new structure integer'])] = row['cpq']
       except KeyError:
          NC_vs_p_vs_cpq[row['NCindex']] = {int(row['new structure integer']): row['cpq']}
elif type_map == 'HPmodel':
   GPmap=np.load('./HP_data/GPmapHP_L25.npy')      
   K, L = int(GPmap.shape[1]), int(GPmap.ndim)
   df = pd.read_csv('./HP_data/HP_NCs_L'+str(L)+'_forsimulation.csv')
   NCindex_list = df['NCindex'].tolist()
   NCindex_vs_structure = {NCindex: structure_int for NCindex, structure_int in zip(df['NCindex'].tolist(), df['structure integer'].tolist())}
   NCindex_arrayRNA = NC.find_all_NCs_parallel(GPmap, filename='./HP_data/HP_NCindexArray_L'+str(L)+'.npy')
   ####################################################################################################
   print('load cpq values', flush=True)
   ####################################################################################################
   filename_cpq = './HP_data/HP_NCs_cqp_data_L'+str(L)+'.csv'
   df_cpq, NC_vs_p_vs_cpq = pd.read_csv(filename_cpq), {}
   for i, row in df_cpq.iterrows():
       try:
          NC_vs_p_vs_cpq[row['NCindex']][int(row['new structure integer'])] = row['cpq']
       except KeyError:
          NC_vs_p_vs_cpq[row['NCindex']] = {int(row['new structure integer']): row['cpq']} 
elif type_map == 'biomorphs':
   str_GPmapparams = '3_1_8_30_5'
   df = pd.read_csv('./biomorph_data/biomorph_NCs_'+str_GPmapparams+'_forsimulation.csv')
   NCindex_list = df['NCindex'].tolist()
   NCindex_vs_structure = {NCindex: structure_int for NCindex, structure_int in zip(df['NCindex'].tolist(), df['structure integer'].tolist())} 
   GPmap = np.load('./biomorph_data/phenotypes' + str_GPmapparams + '.npy')      
   NCindex_arrayRNA = np.load('./biomorph_data/neutral_components'+str_GPmapparams+'.npy')
   ####################################################################################################
   print('load cpq values', flush=True)
   ####################################################################################################
   filename_cpq = './biomorph_data/biomorph_NCs_cqp_data'+str_GPmapparams+'.csv'
   df_cpq, NC_vs_p_vs_cpq = pd.read_csv(filename_cpq), {}
   for i, row in df_cpq.iterrows():
      if row['NCindex'] in NCindex_list:
         try:
             NC_vs_p_vs_cpq[row['NCindex']][int(row['new structure integer'])] = row['cpq']
         except KeyError:
             NC_vs_p_vs_cpq[row['NCindex']] = {int(row['new structure integer']): row['cpq']}  

####################################################################################################
print('run', flush=True)
####################################################################################################
for NCindex in NCindex_list:        
   string_saving_NC = string_saving +'NCindex'+str(NCindex) +'_'
   p0 = NCindex_vs_structure[NCindex] #GPmapRNA[NC.find_index(NCindex, NCindex_arrayRNA)]
   phi_pq_local = NC_vs_p_vs_cpq[NCindex]
   rare_phenos = np.array([p for p in phi_pq_local.keys() if p!=p0 and rare_pheno_min_f <= phi_pq_local[p]<= rare_pheno_max_f], dtype='int')
   print('rare_phenos', rare_phenos)
   filename_saving = './evolution_data/'+type_map+'_times_p_found'+string_saving_NC
   ##### check if file exists
   file_exists = True
   for ph in [int(ph) for ph in rare_phenos] + ['robustness_list', 'genotype_list']: 
      if not isfile(filename_saving+str(ph)+'.npy'):
        file_exists = False
   if file_exists:
      continue
   ###
   if type_map != 'randommap' and type_map != 'biomorphs':
      ################################################################################################################################################################################
      print('input: GP map properties', flush=True)
      ################################################################################################################################################################################
      if type_map == 'HPmodel':
         GPmap=np.load('./HP_data/GPmapHP_L25.npy') 
      elif set_other_NCs_in_neutral_set_to_zero:
         GPmap = np.load('./singleNC_data/'+type_map+'_GPmap_L'+str(L)+'NCindex'+str(NCindex)+'_other_NCs_in_neutral_set_removed.npy')
      elif type_map == 'RNA':
         GPmap = np.load('./'+type_map+'_data/'+type_map+'_GPmap_L'+str(L)+'.npy')
      else:
         GPmap = np.load('./'+type_map+'_data/'+type_map+'_GPmap_L'+str(L)+'NCindex'+str(NCindex)+'.npy')
      ################################################################################################################################################################################
      print('initial conditions', flush=True)
      ################################################################################################################################################################################
      V = np.argwhere(NCindex_arrayRNA==NCindex)
      randomindex=np.random.randint(0, high=V.shape[0]-1) #choose random startgene from NC V
      startgene=tuple([int(i) for i in V[randomindex,:]])
      assert GPmap[startgene]==p0
      ################################################################################################################################################################################
      print('run Wright Fisher', flush=True)
      ################################################################################################################################################################################
      times_p_found, times_p_found_corresp_p, robustness_list, genotype_list = run_WrightFisher(T, N, mu, startgene, rare_phenos, GPmap, p0)
      np.save(filename_saving+'robustness_list'+'.npy', robustness_list)
      np.save(filename_saving+'genotype_list'+'.npy', genotype_list)
      for ph in rare_phenos: 
          times_p_found_for_p = [t for t_i, t in enumerate(times_p_found) if abs(times_p_found_corresp_p[t_i] - ph) < 0.001]       
          np.save(filename_saving+str(ph)+'.npy', times_p_found_for_p)

   elif type_map == 'randommap':   
      ################################################################################################################################################################################
      print('input: GP map properties', flush=True)
      ################################################################################################################################################################################
      GPmap=np.load('./randommap_data/randommapGPmap_L'+str(L)+'NCindex'+str(NCindex)+'.npy')
      ################################################################################################################################################################################
      print('initial conditions', flush=True)
      ################################################################################################################################################################################
      V = np.argwhere(GPmap==p0)
      randomindex=np.random.randint(0, high=V.shape[0]-1) #choose random startgene from NC V
      startgene=tuple(V[randomindex,:])
      assert GPmap[startgene]==p0
      ################################################################################################################################################################################
      print('run Wright Fisher', flush=True)
      ################################################################################################################################################################################
      times_p_found, times_p_found_corresp_p, robustness_list, genotype_list = run_WrightFisher(T, N, mu, startgene, rare_phenos, GPmap, p0)
      np.save(filename_saving+'robustness_list'+'.npy', robustness_list)
      np.save(filename_saving+'genotype_list'+'.npy', genotype_list)
      for ph in rare_phenos: 
          times_p_found_for_p = [t for t_i, t in enumerate(times_p_found) if times_p_found_corresp_p[t_i] == ph]        
          np.save(filename_saving+str(ph)+'.npy', times_p_found_for_p)
   elif type_map == 'biomorphs':   
      ################################################################################################################################################################################
      print('input: GP map properties', flush=True)
      ################################################################################################################################################################################
      GPmap = np.load('./biomorph_data/phenotypes' + str_GPmapparams + '.npy')
      ################################################################################################################################################################################
      print('initial conditions', flush=True)
      ################################################################################################################################################################################
      V = np.argwhere(NCindex_arrayRNA==NCindex)
      randomindex=np.random.randint(0, high=V.shape[0]-1) #choose random startgene from NC V
      startgene=tuple([int(i) for i in V[randomindex,:]])
      assert GPmap[startgene]==p0
      ################################################################################################################################################################################
      print('run Wright Fisher', flush=True)
      ################################################################################################################################################################################
      times_p_found, times_p_found_corresp_p, genotype_list = run_WrightFisher_biomorphs(T, N, mu, startgene, rare_phenos, GPmap, p0)
      np.save(filename_saving+'genotype_list'+'.npy', genotype_list)
      for ph in rare_phenos: 
          times_p_found_for_p = [t for t_i, t in enumerate(times_p_found) if abs(times_p_found_corresp_p[t_i] - ph) < 0.001]       
          np.save(filename_saving+str(ph)+'.npy', times_p_found_for_p)

   