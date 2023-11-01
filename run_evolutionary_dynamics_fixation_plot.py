#!/usr/bin/env python3
import numpy as np
import sys
from evolution_functions.evolutionary_dynamics_function_fixation_numba import run_fixation_n_times, find_startgeno_randommap
from evolution_functions.fixation_nullmodel_numba import run_fixation_n_times_meanfield
from NC_properties.RNA_seq_structure_functions import sequence_str_to_int, isundefinedpheno
from numba import jit
import NC_properties.general_functions as g
from os.path import isfile
import pandas as pd

type_map=sys.argv[1]
assert type_map in ['RNA', 'randommap', 'topology_map', 'community_map', 'meanfield']
N, mu, repetitions = int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
s1, s2 = float(sys.argv[5]), float(sys.argv[6])
startgene_str = sys.argv[7]
p0, p1, p2 = int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10])
set_other_NCs_in_neutral_set_to_zero = int(sys.argv[11])
T = 10**7
################################################################################################################################################################################
## only run if s2>s1 or f2 == 0
################################################################################################################################################################################
if s1 - 10**(-3) >= s2 and abs(s2 + 1) > 0.01:
   raise RuntimeError('This script is programmed to stop unless we have s2>s1 or f2 == 0.')

################################################################################################################################################################################
## input: population properties
################################################################################################################################################################################
string_saving = 'twopeak_N'+g.format_integer_filename(N)+'_T'+g.format_integer_filename(T)+'_mu'+g.format_float_filename(mu)+'_r'+g.format_integer_filename(repetitions)
if set_other_NCs_in_neutral_set_to_zero:
   string_saving += 'otherNCszero'
   print(string_saving)
################################################################################################################################################################################
## input: GP map properties
################################################################################################################################################################################
string_saving += 'p1_'+g.format_integer_filename(p1)+'p2_'+g.format_integer_filename(p2) + startgene_str + 's1_'+g.format_float_filename(s1)+'s2_'+g.format_float_filename(s2)
print(type_map, string_saving)
f0 = 1.0
if abs(s1 + 1) > 10**(-3):
   f1 = f0 + s1
else:
   f1 = 0 # zero-peaked case
if abs(s2 + 1) > 10**(-3):
   f2 = f0 + s2
else:
   f2 = 0
L, K = 12, 4
NCindex_arrayRNA = np.load('./RNA_data/RNA_NCindexArray_L'+str(L)+'.npy')
NCindex = NCindex_arrayRNA[sequence_str_to_int(startgene_str)]
filename_discoverytime = './evolution_data/'+type_map+'_fixation_discovery_time_allruns_array'+string_saving+'.npy'
####################################################################################################
##input: GP map properties
####################################################################################################
if type_map!='randommap' and type_map != 'meanfield' and not isfile(filename_discoverytime):
   if set_other_NCs_in_neutral_set_to_zero:
      GPmapRNA = np.load('./singleNC_data/'+type_map+'_GPmap_L'+str(L)+'NCindex'+str(NCindex)+'_other_NCs_in_neutral_set_removed.npy')
   elif type_map == 'RNA':
      GPmapRNA = np.load('./'+type_map+'_data/'+type_map+'_GPmap_L'+str(L)+'.npy')
   else:
      GPmapRNA=np.load('./'+type_map+'_data/'+type_map+'_GPmap_L'+str(L)+'NCindex'+str(NCindex)+'.npy')
   ################################################################################################################################################################################
   ##load GP map properties
   ################################################################################################################################################################################   
   V = np.argwhere(NCindex_arrayRNA==NCindex)
   ################################################################################################################################################################################
   ### initial conditions
   ################################################################################################################################################################################
   startgene_list=[]
   for run_number in range(repetitions): 
      randomindex=np.random.randint(0, high=V.shape[0]-1) #choose random startgene from NC V
      startgene=tuple(V[randomindex,:])
      assert GPmapRNA[startgene]==p0
      startgene_list.append(startgene)
   del V
   ################################################################################################################################################################################
   ### run simulation and save
   ################################################################################################################################################################################

   discovery_time_array, fixation_time_array, fixating_pheno_array, run_vs_times_appeared_p1, run_vs_times_appeared_p2, run_vs_genotype_list = run_fixation_n_times(T, N, mu, startgene_list, p0, p1, p2, GPmapRNA, f0, f1, f2, repetitions)

          
   np.save(filename_discoverytime, discovery_time_array)
   np.save('./evolution_data/'+type_map+'_fixation_time_allruns_array'+string_saving+'.npy', fixation_time_array)
   np.save('./evolution_data/'+type_map+'_fixating_pheno_array'+string_saving+'.npy', fixating_pheno_array)
   for run in range(repetitions):
      np.save('./evolution_data/'+type_map+'_fixation_times_new_p1_'+string_saving+'run'+str(run)+'.npy', run_vs_times_appeared_p1[run])
      np.save('./evolution_data/'+type_map+'_fixation_times_new_p2_'+string_saving+'run'+str(run)+'.npy', run_vs_times_appeared_p2[run])
      np.save('./evolution_data/'+type_map+'_fixation_genotype_path_'+string_saving+'run'+str(run)+'.npy', run_vs_genotype_list[run])

elif type_map=='randommap' and not isfile(filename_discoverytime):   
   ################################################################################################################################################################################
   ##input: GP map properties
   ################################################################################################################################################################################
   GPmap=np.load('./randommap_data/randommapGPmap_L'+str(L)+'NCindex'+str(NCindex)+'.npy')
   ################################################################################################################################################################################
   ### initial conditions
   ################################################################################################################################################################################
   startgene_list=[find_startgeno_randommap(GPmap, p0) for run_number in range(repetitions)]

   ################################################################################################################################################################################
   ### run simulation and save
   ################################################################################################################################################################################

   discovery_time_array, fixation_time_array, fixating_pheno_array, run_vs_times_appeared_p1, run_vs_times_appeared_p2, run_vs_genotype_list = run_fixation_n_times(T, N, mu, startgene_list, p0, p1, p2, GPmap, f0, f1, f2, repetitions)

          
   np.save(filename_discoverytime, discovery_time_array)
   np.save('./evolution_data/'+type_map+'_fixation_time_allruns_array'+string_saving+'.npy', fixation_time_array)
   np.save('./evolution_data/'+type_map+'_fixating_pheno_array'+string_saving+'.npy', fixating_pheno_array)
   for run in range(repetitions):
      np.save('./evolution_data/'+type_map+'_fixation_times_new_p1_'+string_saving+'run'+str(run)+'.npy', run_vs_times_appeared_p1[run])
      np.save('./evolution_data/'+type_map+'_fixation_times_new_p2_'+string_saving+'run'+str(run)+'.npy', run_vs_times_appeared_p2[run])
      np.save('./evolution_data/'+type_map+'_fixation_genotype_path_'+string_saving+'run'+str(run)+'.npy', run_vs_genotype_list[run])

 
elif type_map=='meanfield' and not isfile(filename_discoverytime):  
   ################################################################################################################################################################################
   ##input: NC properties
   ################################################################################################################################################################################
   GPmapRNA = np.load('./RNA_data/'+'RNA'+'_GPmap_L'+str(L)+'.npy')
   filename_cpq = './RNA_data/RNA_NCs_cqp_data_L'+str(L)+'.csv'
   filename_rho = './RNA_data/RNA_NCs_L'+str(L)+'.csv'
   df_cpq, df_rho, NC_vs_p_vs_cpq = pd.read_csv(filename_cpq), pd.read_csv(filename_rho), {}
   for i, row in df_cpq.iterrows():
       try:
          NC_vs_p_vs_cpq[row['NCindex']][int(row['new structure integer'])] = row['cpq']
       except KeyError:
          NC_vs_p_vs_cpq[row['NCindex']] = {int(row['new structure integer']): row['cpq']}
   NCindex_vs_rho = {row['NCindex']: row['rho'] for i, row in df_rho.iterrows()}
   ################################################################################################################################################################################
   ##input: phipq
   ################################################################################################################################################################################
   filename_phipq = './RNA_data/RNA_phiqp_and_rho_data_L'+str(L)+'.csv'
   df_phipq, q_vs_p_vs_phipq_and_rho = pd.read_csv(filename_phipq), {}
   for i, row in df_phipq.iterrows():
       try:
          q_vs_p_vs_phipq_and_rho[row['old structure integer']][int(row['new structure integer'])] = row['phipq']
       except KeyError:
          q_vs_p_vs_phipq_and_rho[row['old structure integer']] = {int(row['new structure integer']): row['phipq']}
   ################################################################################################################################################################################
   ##input: GP map properties
   ################################################################################################################################################################################
   mean_field_rates = np.array([[NCindex_vs_rho[NCindex], NC_vs_p_vs_cpq[NCindex][p1], NC_vs_p_vs_cpq[NCindex][p2]],
                       [q_vs_p_vs_phipq_and_rho[p1][p0], q_vs_p_vs_phipq_and_rho[p1][p1], q_vs_p_vs_phipq_and_rho[p1][p2]],
                       [q_vs_p_vs_phipq_and_rho[p2][p0], q_vs_p_vs_phipq_and_rho[p2][p1], q_vs_p_vs_phipq_and_rho[p2][p2]]])
   ################################################################################################################################################################################
   ### run simulation and save
   ################################################################################################################################################################################
   assert isundefinedpheno(0) #this is the undefined phenotype in the mean-field model
   discovery_time_array, fixation_time_array, fixating_pheno_array, run_vs_times_appeared_p1, run_vs_times_appeared_p2 = run_fixation_n_times_meanfield(T, N, mu * L, p0, p1, p2, mean_field_rates, f0, f1, f2, repetitions)

          
   np.save(filename_discoverytime, discovery_time_array)
   np.save('./evolution_data/'+type_map+'_fixation_time_allruns_array'+string_saving+'.npy', fixation_time_array)
   np.save('./evolution_data/'+type_map+'_fixating_pheno_array'+string_saving+'.npy', fixating_pheno_array)
   for run in range(repetitions):
      np.save('./evolution_data/'+type_map+'_fixation_times_new_p1_'+string_saving+'run'+str(run)+'.npy', run_vs_times_appeared_p1[run])
      np.save('./evolution_data/'+type_map+'_fixation_times_new_p2_'+string_saving+'run'+str(run)+'.npy', run_vs_times_appeared_p2[run])

 




