#!/usr/bin/env python3
import numpy as np
import sys
print('python version number', sys.version_info, flush=True)
from evolutionary_dynamics_function_long_seq import *
import NC_properties.neutral_component as NC
import pandas as pd
import NC_properties.general_functions as g
from os.path import isfile
from NC_properties.RNA_seq_structure_functions import  dotbracket_to_int, sequence_int_to_str, sequence_str_to_int



################################################################################################################################################################################

type_map = 'RNA'
################################################################################################################################################################################
print('population parameters', flush=True)
################################################################################################################################################################################
N, mu = int(sys.argv[1]), float(sys.argv[2])  #5 * 10**7
T = 10**7
print('N, mu, T', N, mu, T)
################################################################################################################################################################################
L = 30 #20
required_sample_size, length_per_walk, every_Nth_seq = 5, 10**5, 20
filename_structure_list = './data_long_seq/structure_list_L' + str(L) + '_sample' + str(required_sample_size) + '.csv'
phipq_filename_sampling = './data_long_seq/phi_pq_data' + str(L) + '_sample' + str(required_sample_size) +'_rwparams' + str(length_per_walk) + '_'+str(every_Nth_seq) + '.csv'
################################################################################################################################################################################
rare_pheno_min_f, rare_pheno_max_f = 0.001, 0.05
################################################################################################################################################################################
################################################################################################################################################################################
string_saving = 'neutral_N'+g.format_integer_filename(N)+'_T'+g.format_integer_filename(T)+'_mu'+g.format_float_filename(mu)+'minf'+g.format_float_filename(rare_pheno_min_f)+'maxf'+g.format_float_filename(rare_pheno_max_f)
####################################################################################################
print('load GP map properties', flush=True)
####################################################################################################
df_phipq, p_vs_phipq, df_structures = pd.read_csv(phipq_filename_sampling), {}, pd.read_csv(filename_structure_list)
for i, row in df_phipq.iterrows():
   try:
      p_vs_phipq[row['structure 1']][row['structure 2']] = row['phi']
   except KeyError:
      p_vs_phipq[row['structure 1']] = {row['structure 2']: row['phi']}

####################################################################################################
print('run', flush=True)
####################################################################################################
for p0, startgene in zip(df_structures['structure'].tolist()[:], df_structures['start sequence'].tolist()[:]): 
   print(p0, startgene)       
   string_saving_NC = string_saving +'structure'+ str(dotbracket_to_int(p0)) +'_' + startgene + '_'
   rare_phenos = [dotbracket_to_int(ph) for ph, phipq in p_vs_phipq[p0].items() if rare_pheno_min_f <= phipq <= rare_pheno_max_f and ph != p0]
   filename_saving = './data_long_seq/'+type_map+'_times_p_found'+string_saving_NC
   ##### check if file exists
   file_exists = True
   for ph in [ph for ph in rare_phenos] + [ 'genotype_list']: 
      if not isfile(filename_saving+str(ph)+'.npy'):
        file_exists = False
   if file_exists:
      continue
   ################################################################################################################################################################################
   print('run Wright Fisher', flush=True)
   ################################################################################################################################################################################
   times_p_found, times_p_found_corresp_p, genotype_list = run_WrightFisher(T, N, mu, sequence_str_to_int(startgene), rare_phenos, L, dotbracket_to_int(p0))
   np.save(filename_saving+'genotype_list'+'.npy', genotype_list)
   for ph in rare_phenos: 
       times_p_found_for_p = [t for t_i, t in enumerate(times_p_found) if abs(times_p_found_corresp_p[t_i] - ph) < 0.001]       
       np.save(filename_saving+str(ph)+'.npy', times_p_found_for_p)

