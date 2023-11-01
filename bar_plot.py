#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.stats import poisson, binom
import pandas as pd
import NC_properties.general_functions as g
import seaborn as sns
from bar_plot_functions import *
from os.path import isfile
from NC_properties.RNA_seq_structure_functions import get_dotbracket_from_int, isundefinedpheno
import RNA
import sys
import phenotypes_to_draw as param

def rough_approx(x, phipq, tau, t_pg):
   return sum([poisson.pmf(x,tau/t_pg * n1) * binom.pmf(n1, (K-1)*L, phipq, loc=0) for n1 in range(0, (K-1)*L + 1)])


plot_types = ['randommap', 'topology_map', 'community_map', 'RNA']
plot_type_vs_title = {'randommap': 'random GP map', 'topology_map': 'topology GP map', 'community_map': 'community GP map', 'RNA':'full RNA GP Map'}
################################################################################################################################################################################
## input: population properties
################################################################################################################################################################################
N, mu, T = int(sys.argv[1]), float(sys.argv[2]), 10**7
L, K = 12, 4
set_other_NCs_in_neutral_set_to_zero = 1
rare_pheno_min_f_data, rare_pheno_max_f_data = 0.001, 0.1
rare_pheno_min_f_plots, rare_pheno_max_f_plots = 0.01, 0.05
string_saving = 'neutral_N'+g.format_integer_filename(N)+'_T'+g.format_integer_filename(T)+'_mu'+g.format_float_filename(mu)+'minf'+g.format_float_filename(rare_pheno_min_f_data)+'maxf'+g.format_float_filename(rare_pheno_max_f_data)
if set_other_NCs_in_neutral_set_to_zero:
   string_saving += 'otherNCszero'
logscaley = True
####################################################################################################
##input: GP map properties
####################################################################################################
df = pd.read_csv('./RNA_data/RNA_NCs_L'+str(L)+'_forrandommap.csv')
NCindex_list = df['NCindex'].tolist()
GPmapRNA=np.load('./RNA_data/RNA_GPmap_L'+str(L)+'.npy')
NCindex_arrayRNA = np.load('./RNA_data/RNA_NCindexArray_L'+str(L)+'.npy')
colors = {plot_types[3 -N]: color for N, color in enumerate(sns.color_palette('viridis', as_cmap=True)(np.linspace(0, 1.0, 4)))}
####################################################################################################
print('load cpq values', flush=True)
####################################################################################################
filename_cpq = './RNA_data/RNA_NCs_cqp_data_L'+str(L)+'.csv'
df_cpq, NC_vs_p_vs_cpq, NC_vs_p_vs_number_portal_genotypes = pd.read_csv(filename_cpq), {}, {}
for i, row in df_cpq.iterrows():
    try:
       NC_vs_p_vs_cpq[row['NCindex']][int(row['new structure integer'])] = row['cpq']
       NC_vs_p_vs_number_portal_genotypes[row['NCindex']][int(row['new structure integer'])] = row['number portal genotypes']
    except KeyError:
       NC_vs_p_vs_cpq[row['NCindex']] = {int(row['new structure integer']): row['cpq']}
       NC_vs_p_vs_number_portal_genotypes[row['NCindex']] = {int(row['new structure integer']): row['number portal genotypes']}
NCindex_vs_rho = {row['NCindex']: row['rho'] for i, row in df.iterrows()}
NCindex_vs_phenotype = {row['NCindex']: row['structure integer'] for i, row in df.iterrows()}
NCindex_vs_genotype = {row['NCindex']: row['example sequence'] for i, row in df.iterrows()}

####################################################################################################
print('full barplots')
####################################################################################################
NCindex, phenotype = param.NCindex_first_plots, param.pb_first_plots
for rough_approx_plotted in [False]:
   ####################################################################################################
   # load data for NC
   ####################################################################################################
   string_saving_NC = string_saving +'NCindex'+str(NCindex) +'_'
   print(string_saving_NC)
   p0 = NCindex_vs_phenotype[NCindex]
   phi_pq_local =  NC_vs_p_vs_cpq[NCindex]
   rho_p0 = NCindex_vs_rho[NCindex]
   print('NCindex', NCindex)
   #print('phi_pq_local', phi_pq_local)
   rare_phenos = [p for p in phi_pq_local.keys() if p != p0 and rare_pheno_min_f_plots <= phi_pq_local[p]<= rare_pheno_max_f_plots and not isundefinedpheno(p)]
   #print('rare phenotypes', rare_phenos, [phi_pq_local[p] for p in rare_phenos])
   t_neutral = 1/(mu*L*rho_p0)
   tau = 100 * max(int(t_neutral//500), 1) # could have higher tau
   t_pg = float((K-1)/float(N*mu))
   print('neutral fixation time', t_neutral, 'tau', tau, 'mutational neighbourhood timescale', t_pg, 'tau ratio', tau/t_pg)
   #print('expected hamming distance two members', ((L * (K-1)/K)**(-1) + (2 * N * L * mu)**(-1))**(-1))
   #print('expected number of mutants per tau is one p accessible per genotype', tau * mu * L * N/(L * 3))
   #print('fraction at hamming distance one (theory)', fraction_at_Hamming_distance_x_randommap_theory(N, mu, L, rho_p0))
   ###
   RNA.PS_rna_plot(' '*L, get_dotbracket_from_int(p0), './plots_structures/phenotype_NC_'+str(NCindex) +'.ps')
   ####################################################################################################
   # plots
   ####################################################################################################
   print('-> phenotype for plot; phipq', phi_pq_local[phenotype])
   print('-> prob to have 0 or 1 in mutational neighbourhood', [binom.pmf(n_p, (K-1)*L, phi_pq_local[phenotype], loc=0) for n_p in (0, 1)])
   RNA.PS_rna_plot(' '*L, get_dotbracket_from_int(phenotype), './plots_structures/barplot_p1_'+str(phenotype) +'.ps')
   print('NCindex', NCindex, ', ph', phenotype, '\nphi_pq_local[phenotype]', phi_pq_local[phenotype])
   print('NC info', get_dotbracket_from_int(p0), NCindex_vs_genotype[NCindex], ', ph', get_dotbracket_from_int(phenotype))
   print('number of portal genotypes', NC_vs_p_vs_number_portal_genotypes[NCindex][phenotype], 'out of NC size', np.sum(NCindex_arrayRNA == NCindex), '\n---\n')
   f,axarr = plt.subplots(ncols=len(plot_types), figsize=(4*len(plot_types)+1, 6.7), dpi=60, sharex=True)
   max_x = 0
   for plotindex, type_map in enumerate(plot_types):
      times_p_found = np.load('./evolution_data/'+type_map+'_times_p_found'+string_saving_NC+str(phenotype)+'.npy')
      max_x = max(max_x, max(times_p_found))
   for plotindex, type_map in enumerate(plot_types):
      times_p_found = np.load('./evolution_data/'+type_map+'_times_p_found'+string_saving_NC+str(phenotype)+'.npy')
      axarr[plotindex].set_title(plot_type_vs_title[type_map])
      mutants_over_steps_of_tau_p = bar_chart_counts_from_discovery_time_list(times_p_found, tau, T)
      simulation_average_p = np.mean(mutants_over_steps_of_tau_p)
      if len(times_p_found)>50:
         print('expected and simulation average', type_map, round(tau*N*L*mu *phi_pq_local[phenotype], 2), simulation_average_p)
      bin_array=[-0.5+i for i in range(0,int(max(mutants_over_steps_of_tau_p)+2))]
      axarr[plotindex].hist(mutants_over_steps_of_tau_p, bins=bin_array, rwidth=0.9, alpha=0.9, edgecolor=colors[type_map], facecolor=colors[type_map], density=True, zorder=1) 
      axarr[plotindex].set_xlabel(r'number of $p_{b}$ appearing'+'\n'+ 'per ' +str(tau)+r' generations')
      axarr[plotindex].set_xlim(-0.5, 1+max(mutants_over_steps_of_tau_p))
      axarr[plotindex].set_ylabel(r'relative frequency')
      ###################################
      if NCindex == 196 and phenotype == 19529749:
         print('fraction of bins over mu + sigma of Poisson', len([b for b in mutants_over_steps_of_tau_p if b > simulation_average_p + simulation_average_p**0.5])/len(mutants_over_steps_of_tau_p))
         print('fraction of bins under mu - sigma of Poisson', len([b for b in mutants_over_steps_of_tau_p if b < simulation_average_p - simulation_average_p**0.5])/len(mutants_over_steps_of_tau_p))
         print('fraction of bins within mu +- sigma of Poisson', len([b for b in mutants_over_steps_of_tau_p if simulation_average_p - simulation_average_p**0.5 < b < simulation_average_p + simulation_average_p**0.5])/len(mutants_over_steps_of_tau_p))
      #####
      ## draw Poisson distribution and prediction  
      ####  
      x1 = np.arange(0, max(mutants_over_steps_of_tau_p))
      axarr[plotindex].plot(x1, np.array(poisson.pmf(x1,simulation_average_p)), c='grey', linewidth=1, zorder=2) #plot Poisson
      
      prediction=[prob_distribution(M, phi_pq_local[phenotype], tau, mu, N, K, L, rho_p0, t_neutral, polymorphic_correction = True) for M in range(0, max(mutants_over_steps_of_tau_p))]

      axarr[plotindex].plot(list(range(0, max(mutants_over_steps_of_tau_p))), np.array(prediction), c='cyan', linewidth=1, zorder=3) #plot prediction
      if plotindex == 0:   
         for i in range(10):
            if i * tau/t_pg < max_x:
               axarr[plotindex].plot([i * tau/t_pg, i * tau/t_pg], [10/len(times_p_found), 1.5], c='k', linewidth=0.6, ls=':', zorder=0) #plot prediction
      if rough_approx_plotted:
         axarr[plotindex].plot(x1, [rough_approx(x, phi_pq_local[phenotype], tau, t_pg) for x in x1], c='g', linewidth=1, zorder=2, ls=':') #plot Poisson

      del mutants_over_steps_of_tau_p
      if rough_approx_plotted:
         filename_beginning = 'approx_barplot'
      else:
         filename_beginning = 'barplot'
      if logscaley:
         axarr[plotindex].set_yscale('log')
         axarr[plotindex].set_ylim(10/len(times_p_found), 1.5)
         filename = './plots/' +filename_beginning+ 'polymorphic_correction'+string_saving_NC+'ph'+str(phenotype)+'_tau' + str(tau)+ 'logy.png'
      else:
         filename = './plots/'+filename_beginning + 'polymorphic_correction'+string_saving_NC+'ph'+str(phenotype)+'_tau' + str(tau)+'.png'
      f.tight_layout()
      f.savefig(filename, bbox_inches='tight', dpi=300)
      del times_p_found
   print('------------------------------------\n------------------------------------\n\n')
   plt.close('all')
   #del f, axarr
"""
####################################################################################################
print('RNA barplots only')
####################################################################################################
type_map = 'RNA'
for NCindex in NCindex_list:
   ####################################################################################################
   # load data for NC
   ####################################################################################################
   string_saving_NC = string_saving +'NCindex'+str(NCindex) +'_'
   print(string_saving_NC)
   p0 = NCindex_vs_phenotype[NCindex]
   phi_pq_local =  NC_vs_p_vs_cpq[NCindex]
   rho_p0 = NCindex_vs_rho[NCindex]
   print('NCindex', NCindex)
   print('phi_pq_local', phi_pq_local)
   rare_phenos = [p for p in phi_pq_local.keys() if p != p0 and rare_pheno_min_f_plots <= phi_pq_local[p]<= rare_pheno_max_f_plots and not isundefinedpheno(p)]
   print('rare phenotypes', rare_phenos, [phi_pq_local[p] for p in rare_phenos])
   t_neutral = 1/(mu*L*rho_p0)
   tau = 100 * max(int(t_neutral//500), 1) # could have higher tau

   ####################################################################################################
   # plots
   ####################################################################################################
   number_subplots = len(rare_phenos)
   number_subplots += 1 #one axis for initial structure
   ncols = min(6, number_subplots)
   nrows = int(np.ceil(number_subplots/ncols))
   if nrows > 1:
      f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2.6, 1.7 * nrows + 1.7))
   else:
      f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2.6, 1.7 * nrows + 0.5))
   plotindex = 0
   if nrows > 1:
         axindex = tuple((plotindex//ncols, plotindex%ncols))
   else:
         axindex = plotindex
   ax[axindex].text(0.5, 0.5, 'initial structure:\n' + get_dotbracket_from_int(p0),transform=ax[axindex].transAxes, horizontalalignment='center')
   ax[axindex].axis('off')
   for phenotype in rare_phenos: 
      plotindex += 1
      if nrows > 1:
         axindex = tuple((plotindex//ncols, plotindex%ncols))
      else:
         axindex = plotindex
      times_p_found = np.load('./evolution_data/'+type_map+'_times_p_found'+string_saving_NC+str(phenotype)+'.npy')
      ax[axindex].set_title('target structure:\n' + get_dotbracket_from_int(phenotype), size=15)
      mutants_over_steps_of_tau_p = bar_chart_counts_from_discovery_time_list(times_p_found, tau, T)
      simulation_average_p = np.mean(mutants_over_steps_of_tau_p)
      if len(times_p_found)>50:
         print('expected and simulation average', type_map, round(tau*N*L*mu *phi_pq_local[phenotype], 2), simulation_average_p)
      bin_array=[-0.5+i for i in range(0,int(max(mutants_over_steps_of_tau_p)+2))]
      ax[axindex].hist(mutants_over_steps_of_tau_p, bins=bin_array, rwidth=0.9, alpha=0.9, edgecolor=colors[type_map], facecolor=colors[type_map], density=True) 
      ax[axindex].set_xlabel(r'number of $p_{b}$ appearing'+'\n'+ 'per ' +str(tau)+r' generations')
      ax[axindex].set_xlim(-0.5, 1+max(mutants_over_steps_of_tau_p))
      ax[axindex].set_ylabel(r'relative frequency')
      ###################################
      #####
      ## draw Poisson distribution and prediction  
      ####  
      x1 = np.arange(0, max(mutants_over_steps_of_tau_p))
      ax[axindex].plot(x1, np.array(poisson.pmf(x1,simulation_average_p)), c='grey', linewidth=1) #plot Poisson
      prediction=[prob_distribution(M, phi_pq_local[phenotype], tau, mu, N, K, L, rho_p0, t_neutral, polymorphic_correction = True) for M in range(0, max(mutants_over_steps_of_tau_p))]
      ax[axindex].plot(list(range(0, max(mutants_over_steps_of_tau_p))), np.array(prediction), c='cyan', linewidth=1) #plot prediction
      del mutants_over_steps_of_tau_p
      if logscaley:
         ax[axindex].set_yscale('log')
         ax[axindex].set_ylim(10/len(times_p_found), 1.5)
   if logscaley:
      filename = './plots/barplot' + 'polymorphic_correction'+string_saving_NC+'_tau' + str(tau)+ 'logy.png'
   else:
      filename = './plots/barplot' + 'polymorphic_correction'+string_saving_NC+'_tau' + str(tau)+'.png'
   if nrows * ncols > number_subplots:
     for plotindex in range(number_subplots, nrows * ncols):
        ax[plotindex//ncols, plotindex%ncols].axis('off')
   f.tight_layout()
   f.savefig(filename, bbox_inches='tight', dpi=300)
   print('------------------------------------\n------------------------------------\n\n')
   plt.close('all')
####################################################################################################
print('compare times to mean-field approx')
####################################################################################################

for NCindex in NCindex_list:
   p0 = NCindex_vs_phenotype[NCindex]
   phi_pq_local =  NC_vs_p_vs_cpq[NCindex]
   rare_phenos = [p for p in phi_pq_local.keys() if p!=p0 and rare_pheno_min_f_data <= phi_pq_local[p]<= rare_pheno_max_f_data and not isundefinedpheno(p)]
   string_saving_NC = string_saving +'NCindex'+str(NCindex) +'_'
   for plot_type in ['time', 'rate']:
      ####################################################################################################
      print('plots for', NCindex)
      ####################################################################################################
      f,ax=plt.subplots(figsize=(4,3), dpi=60)
      for phenoindex, phenotype in enumerate(rare_phenos):
         for typemapindex, type_map in enumerate(plot_types):
            expected_timescale = (N * mu * L * phi_pq_local[phenotype])**(-1)
            times_p_found = np.load('./evolution_data/'+type_map+'_times_p_found'+string_saving_NC+str(phenotype)+'.npy')
            if plot_type == 'time':
               inter_disc_time_list = [times_p_found[i] - times_p_found[i-1] for i in range(1, len(times_p_found)) if times_p_found[i-1]>2*N ]
               ax.scatter([expected_timescale,], [np.mean(inter_disc_time_list), ], s=3, alpha=0.7, c = [colors[type_map],])
            elif plot_type == 'rate':
               number_found = len([t for t in times_p_found])
               expected_number_found = N * mu * L * phi_pq_local[phenotype] * T
               ax.scatter([expected_number_found,], [number_found, ], s=3, alpha=0.7, c = [colors[type_map],])
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
      f.savefig('./plots/mean_'+plot_type+'_plots'+string_saving_NC[:-1]+'.png', bbox_inches='tight', dpi=200)"""