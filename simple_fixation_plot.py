#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import sys
import matplotlib.pyplot as plt
import pandas as pd
from NC_properties.RNA_seq_structure_functions import sequence_str_to_int, get_dotbracket_from_int
import NC_properties.general_functions as g
import seaborn as sns
from fixation_time_functions import *
import RNA
import phenotypes_to_draw as param



plot_types = ['meanfield', 'randommap', 'topology_map', 'community_map', 'RNA']
################################################################################################################################################################################
## input: population properties
################################################################################################################################################################################
N, mu, repetitions = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
#startgene_str = sys.argv[4]
#p0, p1, p2 = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
startgene_str, p0, p1, p2 = param.seq_firstplots, param.pg_first_plots, param.pr_first_plots, 27788608
T = 10**7
s2 = -1
s1_list = [0.005, 0.01, 0.03, 0.07, 0.09, 0.15, 0.27, 0.52]
####################################################################################################
##input: GP map properties
####################################################################################################
L = len(startgene_str)
K = 4
GPmapRNA=np.load('./RNA_data/RNA_GPmap_L'+str(L)+'.npy')
NCindex_arrayRNA = np.load('./RNA_data/RNA_NCindexArray_L'+str(L)+'.npy')
NCindex = NCindex_arrayRNA[sequence_str_to_int(startgene_str)]
colors = {plot_types[4 -N]: color for N, color in enumerate(sns.color_palette('viridis', as_cmap=True)(np.linspace(0, 1.0, 4)))}
colors['meanfield'] = 'grey'
label_vs_title = {'RNA': 'full RNA GP Map', 'randommap': 'random GP map', 'topology_map': 'topology GP map', 'community_map': 'community GP map', 'meanfield': 'average-rate'}
####################################################################################################
print('load cpq values', flush=True)
####################################################################################################
filename_cpq = './RNA_data/RNA_NCs_cqp_data_L'+str(L)+'.csv'
filename_rho = './RNA_data/RNA_NCs_L'+str(L)+'.csv'
df_cpq, df_rho, NC_vs_p_vs_cpq = pd.read_csv(filename_cpq), pd.read_csv(filename_rho), {}
for i, row in df_cpq.iterrows():
    try:
       NC_vs_p_vs_cpq[row['NCindex']][int(row['new structure integer'])] = row['cpq']
    except KeyError:
       NC_vs_p_vs_cpq[row['NCindex']] = {int(row['new structure integer']): row['cpq']}
NCindex_vs_rho = {row['NCindex']: row['rho'] for i, row in df_rho.iterrows()}


####################################################################################################
# expected values
####################################################################################################
phi_pq_local =  NC_vs_p_vs_cpq[NCindex]
rho_p0 = NCindex_vs_rho[NCindex]
t_neutral = 1/(mu*L*rho_p0)
print('p0', get_dotbracket_from_int(p0))
print('p1', get_dotbracket_from_int(p1), NC_vs_p_vs_cpq[NCindex][p1])
for i, p in enumerate([p0, p1]):
   ph_RNA = get_dotbracket_from_int(p)
   filename = './plots_structures/fixation_simple_'+['p0', 'p1'][i]+'_p'+str(p) +'.ps'
   RNA.PS_rna_plot(' '*len(ph_RNA), ph_RNA, filename)
print('expected random map burst size', N/(3*L*rho_p0))
####################################################################################################
####################################################################################################
print('vary s1 - plot')
####################################################################################################
####################################################################################################
function_description, function = 'mean', np.mean
####################################################################################################
# data for plots
####################################################################################################
s1_list_theory = np.linspace(min(s1_list) * 0.5, max(s1_list) * 1.5)
number_found = 0
f,axarr=plt.subplots(figsize=(6, 3))
for mapindex, type_map in enumerate(plot_types):
   mean_time_list = []
   for s1 in s1_list:
      string_saving = 'twopeak_N'+g.format_integer_filename(N)+'_T'+g.format_integer_filename(T)+'_mu'+g.format_float_filename(mu)+'_r'+g.format_integer_filename(repetitions) + 'otherNCszero'
      string_saving += 'p1_'+g.format_integer_filename(p1)+'p2_'+g.format_integer_filename(p2) + startgene_str + 's1_'+g.format_float_filename(s1)+'s2_'+g.format_float_filename(s2)
      try:
         discovery_time_array, fixation_time_array, fixating_pheno_array, run_vs_times_appeared_p1, run_vs_times_appeared_p2, run_vs_genotype_list = g.load_fixation_data(type_map, string_saving, repetitions)
         assert len([p for p in fixating_pheno_array if abs(p-p1) < 0.01]) == len(fixating_pheno_array)
         mean_time_list.append(function(fixation_time_array))
         number_found += 1
      except FileNotFoundError:
         mean_time_list.append(np.nan)
         print('not found', './evolution_data/'+type_map+'_fixation_discovery_time_allruns_array'+string_saving+'.npy' )
   axarr.plot(s1_list, mean_time_list, c= colors[type_map], marker='x', label=label_vs_title[type_map], lw=1)

axarr.plot(s1_list_theory, [const_rate_fixation_time(s1, N, phi_pq_local[p1], L, mu) for s1 in s1_list_theory], 
                 c= 'k', label='theory (average-rate)', ls=':', lw=1) 
axarr.plot(s1_list_theory, [bursty_fixation_time(s1, N, phi_pq_local[p1], L, mu, rho_p0, K, polymorphic_correction=True) for s1 in s1_list_theory], 
                 c= 'cyan', label='theory (random map)', ls=':', lw=1) 
axarr.set_xscale('log')
axarr.set_yscale('log')
axarr.set_xlim(min(s1_list_theory), max(s1_list_theory))
axarr.set_xlabel(r'selective advantage $s_r$')
axarr.set_ylabel(function_description + ' fixation time\n(in generations)')
#axarr.set_xlim(0.5 * min(s1_list), 2 * max(s1_list))
axarr.legend(loc='center left', bbox_to_anchor=(1, 0.5))
f.tight_layout()
f.savefig('./plots/fixation_simple'+ function_description+string_saving.split('s1_')[0]+'_varys1_'+'.png', bbox_inches='tight', dpi=200)
