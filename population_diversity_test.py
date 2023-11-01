#!/usr/bin/env python3
import numpy as np
from evolution_functions.evolutionary_dynamics_function_numba_population_diversity import *
import NC_properties.neutral_component as NC
import pandas as pd
import NC_properties.general_functions as g
from os.path import isfile
from bar_plot_functions import *
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import sys
from scipy.stats import pearsonr, linregress
################################################################################################################################################################################

type_map = 'randommap'
################################################################################################################################################################################
print('population parameters', flush=True)
################################################################################################################################################################################
T = 10**6 #5 * 10**7
set_other_NCs_in_neutral_set_to_zero = True
L, K = 12, 4
####################################################################################################
print('load GP map properties', flush=True)
####################################################################################################
df_NCinfo = pd.read_csv('./RNA_data/RNA_NCs_L'+str(L)+'_forrandommap.csv')
NCindex_list = df_NCinfo['NCindex'].tolist() 
NCindex_vs_structure = {NCindex: structure_int for NCindex, structure_int in zip(df_NCinfo['NCindex'].tolist(), df_NCinfo['structure integer'].tolist())}
NCindex_vs_structure_dotbracket = {NCindex: structure_int for NCindex, structure_int in zip(df_NCinfo['NCindex'].tolist(), df_NCinfo['structure'].tolist())}
NCindex_arrayRNA = np.load('./RNA_data/RNA_NCindexArray_L'+str(L)+'.npy')
NCindex_vs_rho = {row['NCindex']: row['rho'] for i, row in df_NCinfo.iterrows()}
####################################################################################################
print('run', flush=True)
####################################################################################################
N_list = [10, 100, 1000, 2000]
for N in N_list:
  mu_critical = (10 * N)**(-1) # for L approx 10
  for mu in [0.005 * mu_critical, 0.1 * mu_critical, 0.2 * mu_critical, 0.5 * mu_critical]:
    string_saving = 'neutral_N'+g.format_integer_filename(N)+'_T'+g.format_integer_filename(T)+'_mu'+g.format_float_filename(mu)
    if set_other_NCs_in_neutral_set_to_zero:
         string_saving += 'otherNCszero'
    for NCindex in NCindex_list:         
       string_saving_NC = string_saving +'NCindex'+str(NCindex) +'_'
       p0 = NCindex_vs_structure[NCindex] #GPmapRNA[NC.find_index(NCindex, NCindex_arrayRNA)]
       filename_saving = './evolution_data/'+type_map+'_times_p_found'+string_saving_NC
       ##### check if file exists
       if isfile(filename_saving+'fraction_at_most_freq_list'+'.npy'):
         continue
       if type_map=='randommap':   
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
          fraction_at_most_freq_list = run_WrightFisher(T, N, mu, startgene, GPmap, p0)
          np.save(filename_saving+'fraction_at_most_freq_list'+'.npy', fraction_at_most_freq_list)
####################################################################################################
print('plot population diversity against predicted value - all NCs', flush=True)
####################################################################################################
f, ax = plt.subplots(figsize=(6,3))
for NCindex_i, NCindex in enumerate(NCindex_list):
    list_predicted_diversity, list_observed_diversity, list_observed_diversity_std, list_N = [], [], [], []
    for N in N_list:
      mu_critical = (10 * N)**(-1) # for L approx 10
      for mu in [0.005 * mu_critical, 0.1 * mu_critical, 0.2 * mu_critical, 0.5 * mu_critical]:
        string_saving = 'neutral_N'+g.format_integer_filename(N)+'_T'+g.format_integer_filename(T)+'_mu'+g.format_float_filename(mu)
        if set_other_NCs_in_neutral_set_to_zero:
             string_saving += 'otherNCszero'
        filename_saving = './evolution_data/'+type_map+'_times_p_found'+ string_saving +'NCindex'+str(NCindex) +'_'
        fraction_at_most_freq_list = np.load(filename_saving+'fraction_at_most_freq_list'+'.npy')
        list_observed_diversity.append(np.mean(fraction_at_most_freq_list))
        list_observed_diversity_std.append(np.std(fraction_at_most_freq_list))
        list_predicted_diversity.append(fraction_at_Hamming_distance_x_randommap_theory(N, mu, L, NCindex_vs_rho[NCindex]))
        list_N.append(N)   
    s = ax.errorbar(list_predicted_diversity, list_observed_diversity, ms=3, label= NCindex_vs_structure_dotbracket[NCindex], 
                   marker='o', yerr=list_observed_diversity_std, lw=0, elinewidth=0.1)#np.log10(list_N))
    #s = ax.errorbar(list_predicted_diversity, list_observed_diversity, ms=5, label= NCindex_vs_structure_dotbracket[NCindex], marker='o', yerr=list_observed_diversity_std)#np.log10(list_N))

    ax.set_xlabel(r'predicted $f_0$')
    ax.set_ylabel(r'observed $f_0$')
    min_axlimit = 0.8 * min(list_predicted_diversity + list_observed_diversity)
    ax.plot([min_axlimit,1], [min_axlimit, 1], c='k')
ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
f.tight_layout()
f.savefig('./plots/test_polymorphic_prediction'+ type_map  + '.png', dpi=200 )


