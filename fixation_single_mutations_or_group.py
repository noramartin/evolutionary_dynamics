#!/usr/bin/env python3
import numpy as np
import pandas as pd
import NC_properties.general_functions as g
from os.path import isfile
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from multiprocessing import Pool
from functools import partial

def get_new_generation_meanfield_nonumba(P_p_current, P_fitness_current, P_p_new, N, fitnessp0p1 ):
   p0, p1 = 0, 1
   P_list = np.divide(P_fitness_current[:],float(np.sum(P_fitness_current[:])))
   rows = np.random.choice(N, size=N, p=P_list, replace=True) #select N rows that get chosen for the next generation
   for individual in range(N):
      P_p_new[individual] = P_p_current[rows[individual]] #fill in genes for next generation
      #mutate with p=mu
      P_fitness_current[individual] = fitnessp0p1[P_p_new[individual]]
   return P_p_new, P_fitness_current





def run_WrightFisher_fixation_meanfield(rep, N, number_p1, s1):
   p0, p1 = 0, 1
   fitnessp0p1 = {p0: 1, p1: 1 + s1}
   assert 2**32 > N
   # arrays
   #disc_timep1p2 =  np.array([-1, -1], dtype=np.int64),  np.array([-1, -1], dtype=np.int64)
   P_p_current=np.zeros(N, dtype=np.uint32) #array for phenotypes
   P_p_new=np.zeros(N, dtype=np.uint32) #array for phenotypes
   P_fitness_current=np.zeros(N, dtype=np.float32) #array for fitnesses
   # initial conditions
   for individual in range(N):
      P_p_current[individual] = p0
      P_fitness_current[individual] = fitnessp0p1[P_p_current[individual]]
   for individual in range(number_p1):
      P_p_current[individual] = p1
      P_fitness_current[individual] = fitnessp0p1[P_p_current[individual]]
   # evolution 
   while True:
      #choose next generation
      P_p_new, P_fitness_current = get_new_generation_meanfield_nonumba(P_p_current, P_fitness_current, P_p_new, N, fitnessp0p1 )   
      P_p_current[:] = P_p_new[:].copy()
      if (P_p_current == p1).sum()  == N: #fixation
         phenotype_fixation = p1
         break 
      elif (P_p_current == p0).sum() == N: #fixation
         phenotype_fixation = p0
         break       	
   return phenotype_fixation


def run_WrightFisher_fixation_meanfield_successive(rep, N, number_p1, s1):
	list_single_introductions = [run_WrightFisher_fixation_meanfield(rep, N, 1, s1) for i in range(number_p1)]
	if sum(list_single_introductions) > 0:
		return 1 #p1 fixed once or more
	else:
		return 0

################################################################################################################################################################################
## input: population properties
################################################################################################################################################################################
repetitions = 10**5 # 10**5
N_list = [20, 50, 100, 250, 500, 1000]#, 2000]
s1_list = [0.005, 0.01, 0.02, 0.04, 0.08]
number_p1 = int(round(1/max(s1_list)))
type_introduction_list = ['single_introduction', 'one_by_one']
####################################################################################################
print('collect data', flush=True)
####################################################################################################
for N in N_list:
  for s1 in s1_list:
    string_saving = 'testfix_N'+g.format_integer_filename(N) + '_repetitions' \
                + g.format_integer_filename(repetitions) + '_s1_'+g.format_float_filename(s1) + '_number_p1_'+g.format_integer_filename(number_p1)
    for type_introduction in type_introduction_list:
       filename_saving = './evolution_data/'+'_times_p_found'+string_saving + type_introduction + 'successful_fixations'+'.npy'
       ##### check if file exists
       if isfile(filename_saving):
         continue
       ################################################################################################################################################################################
       print('run', 'N, s1, type_introduction', N, s1, type_introduction, flush=True)
       ################################################################################################################################################################################
       if type_introduction == 'single_introduction':
       	   #result_array = np.array([run_WrightFisher_fixation_meanfield(rep, N, number_p1, s1) for rep in range(repetitions)])
       	   function_with_params = partial(run_WrightFisher_fixation_meanfield, N=N, number_p1=number_p1, s1=s1)
       	   #with Pool(10) as p:
       	   #	  result_array = p.map(function_with_params, np.arange(repetitions))
       elif type_introduction == 'one_by_one':
       	   #result_array = np.array([run_WrightFisher_fixation_meanfield_successive(rep, N, number_p1, s1) for rep in range(repetitions)])
       	   function_with_params = partial(run_WrightFisher_fixation_meanfield_successive, N=N, number_p1=number_p1, s1=s1)
       	   #with Pool(10) as p:
       	   #	  result_array = p.map(function_with_params, np.arange(repetitions))
       result_array = [function_with_params(r) for r in range(repetitions)]
       np.save(filename_saving, result_array)
          

####################################################################################################
print('load data', flush=True)
####################################################################################################
type_introduction_vs_prob_matrix = {type_introduction: np.zeros((len(N_list), len(s1_list))) for type_introduction in type_introduction_list}
N_times_s_matrix, s_matrix = np.zeros((len(N_list), len(s1_list))), np.zeros((len(N_list), len(s1_list)))
for Nindex, N in enumerate(N_list):
  for s1index, s1 in enumerate(s1_list):
    string_saving = 'testfix_N'+g.format_integer_filename(N) + '_repetitions' \
                + g.format_integer_filename(repetitions) + '_s1_'+g.format_float_filename(s1) + '_number_p1_'+g.format_integer_filename(number_p1)
    for type_introduction in ['single_introduction', 'one_by_one']:
       filename_saving = './evolution_data/'+'_times_p_found'+string_saving + type_introduction + 'successful_fixations'+'.npy'
       try:
          result_array = np.load(filename_saving)
       except IOError:
         print('file not found', filename_saving)
       type_introduction_vs_prob_matrix[type_introduction][Nindex, s1index] = np.mean(result_array)
    N_times_s_matrix[Nindex, s1index] = N * s1
    s_matrix[Nindex, s1index] = s1


####################################################################################################
print('plot data as scatterplot', flush=True)
####################################################################################################
f, ax = plt.subplots(figsize=(4,3))
ax.plot([0,3], [0,3], c='k', lw=0.5)
s = ax.scatter(type_introduction_vs_prob_matrix['single_introduction'].flatten(), type_introduction_vs_prob_matrix['one_by_one'].flatten(), 
               c=N_times_s_matrix.flatten(), s=15, norm=colors.LogNorm(vmin=np.amin(N_times_s_matrix), vmax=np.amax(N_times_s_matrix)))
ax.set_xlabel('fixation probability when\n'+str(number_p1) + r' $p_1$ mutants'+'\nintroduced at once')
ax.set_ylabel('fixation probability when\n'+str(number_p1) + r' $p_1$ mutants'+'\nintroduced one-by-one')
ax.set_xscale('log')
ax.set_yscale('log')
min_prob = min(type_introduction_vs_prob_matrix['single_introduction'].flatten())
ax.set_xlim(0.5 * min_prob, 2)
ax.set_ylim(0.5 * min_prob, 2)
cb = f.colorbar(s, ax=ax)#, norm = colors.LogNorm(), vmin=np.amin(N_times_s_matrix), vmax=np.amax(N_times_s_matrix))
cb.set_label(r'$N \times s_1$')
f.tight_layout()
f.savefig('./plots/concurrent_introduction_or_one_by_one_repetitions' + g.format_integer_filename(repetitions)+'_2.eps', bbox_inches='tight')
del f, ax
plt.close('all')

####################################################################################################
print('plot data as scatterplot - theory', flush=True)
####################################################################################################
f, ax = plt.subplots(figsize=(4,3))
ax.plot([0,3], [0,3], c='k', lw=0.5)
list_subsequent_theory = [1 - (1-((1 - np.exp(-2 * s))/(1-np.exp(-2* sN))))**number_p1 for sN, s in zip(N_times_s_matrix.flatten(), s_matrix.flatten())]
list_single_theory = [(1 - np.exp(-2 * s * number_p1))/(1-np.exp(-2* sN)) for sN, s in zip(N_times_s_matrix.flatten(), s_matrix.flatten())]

s = ax.scatter(list_single_theory, list_subsequent_theory , c=N_times_s_matrix.flatten(), s=15, 
               norm=colors.LogNorm(vmin=np.amin(N_times_s_matrix), vmax=np.amax(N_times_s_matrix)))
ax.set_xlabel('fixation probability (theory) when\n'+str(number_p1) + r' $p_1$ mutants'+'\nintroduced at once')
ax.set_ylabel('fixation probability (theory) when\n'+str(number_p1) + r' $p_1$ mutants'+'\nintroduced one-by-one')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.5 * min_prob, 2)
ax.set_ylim(0.5 * min_prob, 2)
cb = f.colorbar(s, ax=ax)
cb.set_label(r'$N \times s_1$')
f.tight_layout()
f.savefig('./plots/theory_concurrent_introduction_or_one_by_one_repetitions' + g.format_integer_filename(repetitions)+'_2.eps', bbox_inches='tight')
del f, ax
plt.close('all')

