#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import sys
import matplotlib.pyplot as plt
import math
from scipy.stats import poisson
import NC_properties.neutral_component as NC
import pandas as pd
import NC_properties.general_functions as g
from NC_properties.RNA_seq_structure_functions import dotbracket_to_int, sequence_int_to_str
import numpy as np
import seaborn as sns
from bar_plot_functions import *
from os.path import isfile
import sys


################################################################################################################################################################################
## input: population properties
################################################################################################################################################################################
N, mu, T = int(sys.argv[1]), float(sys.argv[2]), 10**7
K = 4
rare_pheno_min_f_data, rare_pheno_max_f_data = 0.001, 0.05
rare_pheno_min_f_plots, rare_pheno_max_f_plots = 0.001, 0.02
string_saving = 'neutral_N'+g.format_integer_filename(N)+'_T'+g.format_integer_filename(T)+'_mu'+g.format_float_filename(mu)+'minf'+g.format_float_filename(rare_pheno_min_f_data)+'maxf'+g.format_float_filename(rare_pheno_max_f_data)
type_map = 'RNA'
colorRNA = [color for N, color in enumerate(sns.color_palette('viridis', as_cmap=True)(np.linspace(0, 1.0, 4)))][0]
####################################################################################################
print('load cpq values', flush=True)
####################################################################################################
L = 30 #20
required_sample_size, length_per_walk, every_Nth_seq = 5, 10**5, 20
filename_structure_list = './data_long_seq/structure_list_L' + str(L) + '_sample' + str(required_sample_size) + '.csv'
phipq_filename_sampling = './data_long_seq/phi_pq_data' + str(L) + '_sample' + str(required_sample_size) +'_rwparams' + str(length_per_walk) + '_'+ str(every_Nth_seq) + '.csv'
######
df_phipq, p_vs_phipq, df_structures = pd.read_csv(phipq_filename_sampling), {}, pd.read_csv(filename_structure_list)
for i, row in df_phipq.iterrows():
   try:
      p_vs_phipq[row['structure 1']][row['structure 2']] = row['phi']
   except KeyError:
      p_vs_phipq[row['structure 1']] = {row['structure 2']: row['phi']}

####################################################################################################

for p0, startgene in zip(df_structures['structure'].tolist(), df_structures['start sequence'].tolist()):          
   string_saving_NC = string_saving +'structure'+ str(dotbracket_to_int(p0)) +'_' + startgene + '_'
   ####################################################################################################
   # load data for NC
   ####################################################################################################
   print(string_saving_NC)
   phi_pq_local =  p_vs_phipq[p0]
   rho_p0 = p_vs_phipq[p0][p0]
   del p_vs_phipq[p0][p0]
   rare_phenos = [p for p in phi_pq_local.keys() if p!=p0 and rare_pheno_min_f_plots <= phi_pq_local[p]<= rare_pheno_max_f_plots]
   print('rare phenotypes', rare_phenos, [phi_pq_local[p] for p in rare_phenos])
   t_neutral = 1/(mu*L*rho_p0)
   tau = 100 * max(int(t_neutral//500), 1) # could have higher tau
   print('neutral fixation time', t_neutral, 'tau', tau)
   ####################################################################################################
   # how many subplots
   ####################################################################################################
   number_subplots = 0
   for phenotype in rare_phenos: 
      if not isfile('./data_long_seq/RNA_times_p_found'+string_saving_NC+str(dotbracket_to_int(phenotype))+'.npy'): 
         continue
      times_p_found = np.load('./data_long_seq/RNA_times_p_found'+string_saving_NC+str(dotbracket_to_int(phenotype))+'.npy')
      mutants_over_steps_of_tau_p = bar_chart_counts_from_discovery_time_list(times_p_found, tau, T)
      if len(times_p_found) < 1000 or np.mean(mutants_over_steps_of_tau_p) <= 2:
         continue   
      number_subplots += 1
   if number_subplots == 0:
      continue
   ####################################################################################################
   # plots
   ####################################################################################################
   ncols = min(5, number_subplots)
   nrows = int(np.ceil(number_subplots/ncols))
   
   f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2.6, 1.7 * nrows + 1.7))
   plotindex = 0
   for phenotype in rare_phenos: 
      print('---\n', 'p0', p0, '\nphi_pq_local[phenotype]', phi_pq_local[phenotype])
      print(phenotype)
      if nrows > 1:
         axindex = tuple((plotindex//ncols, plotindex%ncols))
      else:
         axindex = plotindex
      #f.subplots_adjust(wspace = 0.3)
      if not isfile('./data_long_seq/RNA_times_p_found'+string_saving_NC+str(dotbracket_to_int(phenotype))+'.npy'): 
         print('file not found')
         continue
      times_p_found = np.load('./data_long_seq/RNA_times_p_found'+string_saving_NC+str(dotbracket_to_int(phenotype))+'.npy')
      print(len(times_p_found), 'times found', phenotype, 'from', p0)
      print('expected times found', tau*N*L*mu *phi_pq_local[phenotype])
      mutants_over_steps_of_tau_p = bar_chart_counts_from_discovery_time_list(times_p_found, tau, T)
      if len(times_p_found) < 1000 or np.mean(mutants_over_steps_of_tau_p) <= 2:
         print ('p not found sufficiently')
         continue
      plotindex += 1
      #axarr.set_title('full RNA GP Map', size=15)               
      ax[axindex].set_title(r'from $p_0 = $' + p0+ '\n' + r'to $p_1$ = ' + phenotype, fontsize=10)
      simulation_average_p = np.mean(mutants_over_steps_of_tau_p)
      bin_array=[-0.5+i for i in range(0,int(max(mutants_over_steps_of_tau_p)+2))]
      ax[axindex].hist(mutants_over_steps_of_tau_p, bins=bin_array, rwidth=0.9, alpha=0.75, edgecolor=colorRNA, facecolor=colorRNA, density=True) #plot random map data as hist
      ax[axindex].set_xlabel(r'number of $p_{1}$ appearing'+'\n'+ 'per ' +str(tau)+r' generations')
      ax[axindex].set_xlim(-0.5, 1+max(mutants_over_steps_of_tau_p))
      ax[axindex].set_ylabel(r'relative frequency')
      ###
      x1 = np.arange(0, max(mutants_over_steps_of_tau_p))
      ax[axindex].plot(x1, np.array(poisson.pmf(x1,simulation_average_p)), c='grey', linewidth=1.2) #plot Poisson
      #prediction=[prob_distribution(M, phi_pq_local[phenotype], tau, mu, N, K, L, rho_p0, t_neutral, polymorphic_correction = True) for M in x1]
      #axarr.plot(x1, np.array(prediction), c='k', linewidth=1) #plot prediction
      del mutants_over_steps_of_tau_p#, prediction
      ax[axindex].set_yscale('log')
      ax[axindex].set_ylim(10/len(times_p_found), 1.5)
      del times_p_found
   if nrows * ncols > number_subplots:
     for plotindex in range(number_subplots, nrows * ncols):
        ax[plotindex//ncols, plotindex%ncols].axis('off')
   filename = './plots_long_seq/longseqbarplot'+ 'polymorphic_correction'+string_saving_NC+'_tau' + str(tau)+ 'logy.png'
   f.tight_layout()
   f.savefig(filename, bbox_inches='tight', dpi=300)
   print('------------------------------------\n------------------------------------\n\n')
   plt.close('all')

####################################################################################################
print('compare times to mean-field approx')
####################################################################################################
for p0, startgene in zip(df_structures['structure'].tolist(), df_structures['start sequence'].tolist()): 
   string_saving_NC = string_saving +'structure'+ str(dotbracket_to_int(p0)) +'_' + startgene + '_'
   phi_pq_local =  p_vs_phipq[p0]
   #rho_p0 = p_vs_phipq[p0][p0]
   #del p_vs_phipq[p0][p0]
   rare_phenos = [p for p in phi_pq_local.keys() if p!=p0 and rare_pheno_min_f_data <= phi_pq_local[p]<= rare_pheno_max_f_data]
   for plot_type in ['time', 'rate']:
      ####################################################################################################
      print('plots for', p0)
      ####################################################################################################
      f,ax=plt.subplots(figsize=(4,3), dpi=60)
      for phenoindex, phenotype in enumerate(rare_phenos):
            expected_timescale = (N * mu * L * phi_pq_local[phenotype])**(-1)
            times_p1_found=np.load('./data_long_seq/'+type_map+'_times_p_found'+string_saving_NC +str(dotbracket_to_int(phenotype))+'.npy')
            if plot_type == 'time':
               inter_disc_time_list = [times_p1_found[i] - times_p1_found[i-1] for i in range(1, len(times_p1_found)) ]
               if len(inter_disc_time_list) >= 5:
                  ax.scatter([expected_timescale,], [np.mean(inter_disc_time_list), ], s=3, alpha=0.7, c = ['purple',])
            elif plot_type == 'rate':
               number_found = len([t for t in times_p1_found])
               expected_number_found = N * mu * L * phi_pq_local[phenotype] * T 
               ax.scatter([expected_number_found,], [number_found, ], s=3, alpha=0.7, c = ['purple',])
            ax.set_title('N=' + str(N) + r', $\mu=$' + str(mu))
      ax.set_xscale('log')
      ax.set_yscale('log')
      if plot_type == 'time':
         ax.set_ylabel('mean time\nto next appearance')
         ax.set_xlabel('expected time\nto next appearance')
      else:
         ax.set_ylabel('number found')
         ax.set_xlabel('expected number found')               
      (xmin, xmax), (ymin, ymax) = ax.get_xlim(), ax.get_ylim()
      ax_lims = [min(xmin, ymin), max(xmax, ymax)]
      ax.plot(ax_lims, ax_lims, c='grey')
      ax.set_xlim(ax_lims[0], ax_lims[1])
      ax.set_ylim(ax_lims[0], ax_lims[1])
      f.tight_layout()
      f.savefig('./plots_long_seq/longseqmean_'+plot_type+'_plots'+string_saving+'.png', bbox_inches='tight', dpi=200)

