#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import NC_properties.neutral_component as NC
import pandas as pd
import NC_properties.general_functions as g
from NC_properties.RNA_seq_structure_functions import sequence_str_to_int
import sys
from fixation_time_functions import *
import seaborn as sns
import phenotypes_to_draw as param
#############################################################################################################################
## start
#############################################################################################################################
min_number_fixations = 5 #minimum number of sucessful fixation for a data point to be included
#############################################################################################################################
## input: parameters
#############################################################################################################################
N, mu, repetitions = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
startgene_str, p0, p1, p2 = param.seq_twopeaked, param.p0_twopeaked, param.p1_twopeaked, param.p2_twopeaked

T = 10**7
string_saving = 'twopeak_N'+g.format_integer_filename(N)+'_T'+g.format_integer_filename(T)+'_mu'+g.format_float_filename(mu)+'_r'+g.format_integer_filename(repetitions) + 'otherNCszero'
string_saving += 'p1_'+g.format_integer_filename(p1)+'p2_'+g.format_integer_filename(p2) + startgene_str 
s1 = 0.005
s2_list = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
#############################################################################################################################
## input: phi pq
#############################################################################################################################
L, K = 12, 4
GPmapRNA = np.load('./RNA_data/RNA_GPmap_L'+str(L)+'.npy')
NCindex_arrayRNA = NC.find_all_NCs_parallel(GPmapRNA, filename='./RNA_data/RNA_NCindexArray_L'+str(L)+'.npy')
NCindex = NCindex_arrayRNA[sequence_str_to_int(startgene_str)]
plot_types = ['meanfield', 'randommap', 'topology_map', 'community_map', 'RNA'] + ['theory', 'theory dynamic bursts'] #topology_randomRNAnc
colors = {plot_types[4 -N]: color for N, color in enumerate(sns.color_palette('viridis', as_cmap=True)(np.linspace(0, 1.0, 4)))}
colors['meanfield'] = 'grey'
colors['theory'] = 'k'
colors['theory dynamic bursts'] = 'cyan'
#colors['topology_randomRNAnc'] = 'b'
label_vs_title = {'RNA': 'full RNA GP Map', 'randommap': 'random GP map', 'topology_map': 'topology GP map', 'community_map': 'community GP map', 'meanfield': 'constant-rate', 'theory': 'theory (constant-rate)','theory dynamic bursts': 'theory (random map)'}#, 'topology_randomRNAnc': 'community 2'}

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
phi_pq = NC_vs_p_vs_cpq[NCindex]
rho_p0 = NCindex_vs_rho[NCindex]
print('bias', phi_pq[p1]/phi_pq[p2], 'for phipq', phi_pq[p1], phi_pq[p2])
#############################################################################################################################
## arrays for plot
#############################################################################################################################
type_map_vs_fix = {type_map: [] for type_map in plot_types}
s2_list_for_plot = []
#############################################################################################################################
## read in
#############################################################################################################################
for s2index, s2 in enumerate(s2_list):
     if s1 > s2 - 10.0**(-5):
        continue
     s2_list_for_plot.append(s2)
     string_saving_s = string_saving + 's1_'+g.format_float_filename(s1)+'s2_'+g.format_float_filename(s2)
     for type_map in plot_types[:5]:
        try:
           discovery_list, fixation_time_array, fixation_list, run_vs_times_appeared_p1, run_vs_times_appeared_p2, run_vs_genotype_list = g.load_fixation_data(type_map, string_saving_s, repetitions, load_run_details=False)
           if len([fix for fix in fixation_list if fix > 0.1]) >= min_number_fixations:
              type_map_vs_fix[type_map].append(len([fix for fix in fixation_list if fix==p2])/float(len([fix for fix in fixation_list if fix])))
           else:
              type_map_vs_fix[type_map].append(np.nan)
           print('fraction unsuccessful runs:', len([fix for fix in fixation_list if not fix])/float(len(fixation_list)), '\n')           
           del fixation_list, discovery_list
        except IOError:
           print( 'not found:', type_map, s1, s2, '\n')
           type_map_vs_fix[type_map].append(np.nan)
     ### theory
     fixation1_prob_single_mutant=(1.0-np.exp(-2.0*s1))/(1.0-np.exp(-2.0*N*s1))
     fixation2_prob_single_mutant=(1.0-np.exp(-2.0*s2))/(1.0-np.exp(-2.0*N*s2))
     type_map_vs_fix['theory'].append(1.0/(1 + phi_pq[p1]*fixation1_prob_single_mutant/(phi_pq[p2]*fixation2_prob_single_mutant)))
     fixation1_time_dyn_b = bursty_fixation_time(s1, N, phi_pq[p1], L, mu, rho_p0, K, polymorphic_correction=True)
     fixation2_time_dyn_b = bursty_fixation_time(s2, N, phi_pq[p2], L, mu, rho_p0, K, polymorphic_correction=True)
     type_map_vs_fix['theory dynamic bursts'].append(1.0/(1 + fixation2_time_dyn_b/fixation1_time_dyn_b))
#######################################################################################################################################
print('plot s2/s1 against data - fixed s2')
#######################################################################################################################################

f, ax = plt.subplots(figsize=(5.5, 3))
for mapindex, type_map in enumerate(plot_types):
     if 'theory' not in type_map:
        print(type_map, np.divide(s2_list_for_plot, s1), type_map_vs_fix[type_map])
        ax.plot(np.divide(s2_list_for_plot, s1), type_map_vs_fix[type_map], color= colors[type_map], label=label_vs_title[type_map], ls=':',ms=3, marker='o')
     else:
        ax.plot(np.divide(s2_list_for_plot, s1), type_map_vs_fix[type_map], color= colors[type_map], label=label_vs_title[type_map], ls=':')
ax.set_xscale('log')
x1, x2 = ax.get_xlim()
xticks = [int(x1) + 1, 10, int(x2 * 0.1) * 10]
ax.set_xticks(xticks)
ax.set_xticklabels([str(s) for s  in xticks])
ax.set_xlabel('ratio of selective advantages\n'+r' $s_r/s_f$')
ax.set_ylabel('probability that\nfittest & rarest phenotype '+r'$p_r$'+'\ngoes into fixation first')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
f.tight_layout()
f.savefig('./plots_arrival_frequent/arrival_frequent_2Dplot_'+ string_saving+ '_s1_'+g.format_float_filename(s1) + '.png', bbox_inches='tight', dpi=200)

