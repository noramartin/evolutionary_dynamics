#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.stats import poisson
import pandas as pd
import NC_properties.general_functions as g
import seaborn as sns
from bar_plot_functions import *
from os.path import isfile
from NC_properties.RNA_seq_structure_functions import get_dotbracket_from_int, isundefinedpheno
import RNA
import sys
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from HP_model.HPfunctions import plot_structure
import parameters_HP as paramHP


def plot_phenotype(ph, ax, phenotypes, g9min, color='k', linewidth=3):
   genemax = (phenotypes.shape[0]-1)//2
   g0 = get_startgeno_indices(phenotypes, ph)
   if len(g0) == 8:
      g0 = tuple([g for g in g0] + [0,])
      print('assuming g9 is constant')
   gene = index_to_gene(g0, genemax, g9min)
   draw_biomorph_in_subplot(gene, ax, color=color, linewidth=linewidth)

def index_to_gene(gene_as_index, genemax, g9min):
   difference_index_gene = [genemax,]*8 + [-g9min]
   return tuple([g - difference_index_gene[i] for i, g in enumerate(gene_as_index)])

def directions(gene):
    #convert the gene into directions and the order up to which lines are drawn
    order = gene[8]
    dx = [int(-gene[1]), int(-gene[0]), 0, int(gene[0]), int(gene[1]), int(gene[2]), 0, -gene[2]]
    dy = [int(gene[5]), int(gene[4]), int(gene[3]), int(gene[4]), int(gene[5]), int(gene[6]), int(gene[7]), int(gene[6])]
    return dx, dy, order

def draw_biomorph(xstart, ystart, order, dx, dy, direction, ax, color='k', linewidth=3):
    #draws the lines to make a biomorph
    direction = direction%8
    xnew = xstart + order * dx[direction]
    ynew = ystart + order * dy[direction] 
    ax.plot([xstart, xnew], [ystart, ynew], color=color, linestyle='-', linewidth=linewidth) 
    if order>1:
        draw_biomorph(xnew, ynew, order-1, dx, dy, direction-1, ax, color, linewidth=linewidth)
        draw_biomorph(xnew, ynew, order-1, dx, dy, direction+1, ax, color, linewidth=linewidth)
    return 

def draw_biomorph_in_subplot(gene, ax, color='k', linewidth=3):
    dx, dy, order= directions(gene)
    draw_biomorph(0, 0, order, dx, dy, 2, ax, color=color, linewidth=linewidth)
    ax.set_aspect('equal', 'box')
    ax.axis('off')

def get_startgeno_indices(phenotypes, ph):
   return tuple(np.argwhere(phenotypes == ph)[0])

#type_map = 'HPmodel'
################################################################################################################################################################################
## input: population properties
################################################################################################################################################################################
type_map, N, mu, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), 10**7
rare_pheno_min_f_data, rare_pheno_max_f_data = 0.001, 0.1
rare_pheno_min_f_plots, rare_pheno_max_f_plots = 0.001, 0.05
string_saving = 'neutral_N'+g.format_integer_filename(N)+'_T'+g.format_integer_filename(T)+'_mu'+g.format_float_filename(mu)+'minf'+g.format_float_filename(rare_pheno_min_f_data)+'maxf'+g.format_float_filename(rare_pheno_max_f_data)
if True:
   string_saving += 'otherNCszero'
logscaley = True
if type_map == 'HPmodel':
   GPmap=np.load('./HP_data/GPmapHP_L25.npy')      
   K, L = int(GPmap.shape[1]), int(GPmap.ndim)
   df = pd.read_csv('./HP_data/HP_NCs_L'+str(L)+'_forsimulation.csv')
   NCindex_list = df['NCindex'].tolist()
   ####################################################################################################
   print('load cpq values', flush=True)
   ####################################################################################################
   filename_cpq = './HP_data/HP_NCs_cqp_data_L'+str(L)+'.csv'
elif type_map == 'biomorphs':
   str_GPmapparams = '3_1_8_30_5'
   L = 9
   df = pd.read_csv('./biomorph_data/biomorph_NCs_'+str_GPmapparams+'_forsimulation.csv')
   NCindex_list = df['NCindex'].tolist()
   GPmap = np.load('./biomorph_data/phenotypes' + str_GPmapparams + '.npy')      
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
NCindex_vs_phenotype = {row['NCindex']: row['structure integer'] for i, row in df.iterrows()}
####################################################################################################
for NCindex in NCindex_list:
   ####################################################################################################
   # load data for NC
   ####################################################################################################
   string_saving_NC = string_saving +'NCindex'+str(NCindex) +'_'
   print(string_saving_NC)
   p0 = NCindex_vs_phenotype[NCindex]
   phi_pq_local =  NC_vs_p_vs_cpq[NCindex]
   rho_p0 = phi_pq_local[p0]
   print('NCindex', NCindex)
   #print('phi_pq_local', phi_pq_local)
   rare_phenos = [p for p in phi_pq_local.keys() if p != p0 and rare_pheno_min_f_plots <= phi_pq_local[p]<= rare_pheno_max_f_plots and not (p < 0.5 and type_map == 'HPmodel')]
   print('rare phenotypes', rare_phenos, [phi_pq_local[p] for p in rare_phenos])
   t_neutral = 1/(mu*L*rho_p0)
   tau = 100 * max(int(t_neutral//500), 1) # could have higher tau
   print('neutral fixation time', t_neutral, 'tau', tau)
   ####################################################################################################
   # how many plots
   ####################################################################################################
   number_subplots = 0
   for phenotype in rare_phenos: 
      if not isfile('./evolution_data/'+type_map+'_times_p_found'+string_saving_NC+str(phenotype)+'.npy'): 
         print('file not found')
         continue
      times_p_found = np.load('./evolution_data/'+type_map+'_times_p_found'+string_saving_NC+str(phenotype)+'.npy')
      mutants_over_steps_of_tau_p = bar_chart_counts_from_discovery_time_list(times_p_found, tau, T)
      if len(times_p_found) < 500 or np.mean(mutants_over_steps_of_tau_p) < 2:
         if len(times_p_found) < 500:
            print('not plotting because insufficient data')
         elif np.mean(mutants_over_steps_of_tau_p) < 2:
            print('produced too often for stats')
         continue   
      number_subplots += 1
   if number_subplots == 0:
      continue
   ####################################################################################################
   # plots
   ####################################################################################################
   #if type_map == 'HPmodel':
   number_subplots += 1 #one axis for initial structure
   ncols = min(5, number_subplots)
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
   if type_map == 'HPmodel':
      # plot initial phenotype in first subplot
      plot_structure(paramHP.contact_map_vs_updown[paramHP.contact_map_list[p0 - 1]], ax[axindex], redpointsize=60)
      plotindex += 1
   elif type_map == 'biomorphs':
      # plot initial phenotype in first subplot
      plot_phenotype(p0, ax[axindex], GPmap, g9min=1, color='k', linewidth=3)
      plotindex += 1
   for phenotype in rare_phenos: 
      if nrows > 1:
         axindex = tuple((plotindex//ncols, plotindex%ncols))
      else:
         axindex = plotindex
      if not isfile('./evolution_data/'+type_map+'_times_p_found'+string_saving_NC+str(phenotype)+'.npy'): 
         continue
      times_p_found = np.load('./evolution_data/'+type_map+'_times_p_found'+string_saving_NC+str(phenotype)+'.npy')
      mutants_over_steps_of_tau_p = bar_chart_counts_from_discovery_time_list(times_p_found, tau, T)
      if len(times_p_found) < 500 or np.mean(mutants_over_steps_of_tau_p) < 2:
         continue  
      plotindex += 1
      simulation_average_p = np.mean(mutants_over_steps_of_tau_p)
      if len(times_p_found)>50:
         print('expected and simulation average', type_map, round(tau*N*L*mu *phi_pq_local[phenotype], 2), simulation_average_p)
      bin_array=[-0.5+i for i in range(0,int(max(mutants_over_steps_of_tau_p)+2))]
      ax[axindex].hist(mutants_over_steps_of_tau_p, bins=bin_array, rwidth=0.9, alpha=0.7, edgecolor='green', facecolor='green', density=True) 
      ax[axindex].set_xlabel(r'number of $p_{1}$ appearing'+'\n'+ 'per ' +str(tau)+r' generations')
      ax[axindex].set_xlim(-0.5, 1+max(mutants_over_steps_of_tau_p))
      ax[axindex].set_ylabel(r'relative frequency')
      inset_ax = inset_axes(ax[axindex], width='20%', height='20%', loc=1)
      if type_map == 'HPmodel':        
         plot_structure(paramHP.contact_map_vs_updown[paramHP.contact_map_list[phenotype - 1]], inset_ax, redpointsize=6)
      elif type_map == 'biomorphs':
         plot_phenotype(phenotype, inset_ax, GPmap, g9min=1, color='k', linewidth=0.7)
      ###################################
      if NCindex == 196 and phenotype == 19529749:
         print('fraction of bins over mu + sigma of Poisson', len([b for b in mutants_over_steps_of_tau_p if b > simulation_average_p + simulation_average_p**0.5])/len(mutants_over_steps_of_tau_p))
         print('fraction of bins under mu - sigma of Poisson', len([b for b in mutants_over_steps_of_tau_p if b < simulation_average_p - simulation_average_p**0.5])/len(mutants_over_steps_of_tau_p))
         print('fraction of bins within mu +- sigma of Poisson', len([b for b in mutants_over_steps_of_tau_p if simulation_average_p - simulation_average_p**0.5 < b < simulation_average_p + simulation_average_p**0.5])/len(mutants_over_steps_of_tau_p))
      #####
      ## draw Poisson distribution and prediction  
      ####  
      x1 = np.arange(0, max(mutants_over_steps_of_tau_p))
      ax[axindex].plot(x1, np.array(poisson.pmf(x1,simulation_average_p)), c='grey', linewidth=1) #plot Poisson
      if type_map == 'HPmodel':
         prediction=[prob_distribution(M, phi_pq_local[phenotype], tau, mu, N, K, L, rho_p0, t_neutral, polymorphic_correction = True) for M in range(0, max(mutants_over_steps_of_tau_p))]
         ax[axindex].plot(list(range(0, max(mutants_over_steps_of_tau_p))), np.array(prediction), c='cyan', linewidth=1) #plot prediction
      del mutants_over_steps_of_tau_p
      if logscaley:
         ax[axindex].set_yscale('log')
         ax[axindex].set_ylim(10/len(times_p_found), 1.5)
         filename = './plots/'+type_map+'barplot' + 'polymorphic_correction'+string_saving_NC+'_tau' + str(tau)+ 'logy.png'
      else:
         filename = './plots/'+type_map+'barplot' + 'polymorphic_correction'+string_saving_NC+'_tau' + str(tau)+'.png'
      del times_p_found
   if nrows * ncols > number_subplots:
     for plotindex in range(number_subplots, nrows * ncols):
        ax[plotindex//ncols, plotindex%ncols].axis('off')
   f.tight_layout()
   f.savefig(filename, bbox_inches='tight', dpi=300)    
   print('------------------------------------\n------------------------------------\n\n')
   plt.close('all')
   #del f, axarr

####################################################################################################
print('compare times to mean-field approx --test only')
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
            expected_timescale = (N * mu * L * phi_pq_local[phenotype])**(-1)
            times_p_found = np.load('./evolution_data/'+type_map+'_times_p_found'+string_saving_NC+str(phenotype)+'.npy')
            if plot_type == 'time':
               inter_disc_time_list = [times_p_found[i] - times_p_found[i-1] for i in range(1, len(times_p_found)) if times_p_found[i-1]>2*N ]
               ax.scatter([expected_timescale,], [np.mean(inter_disc_time_list), ], s=3, alpha=0.7, c = ['b',])
            elif plot_type == 'rate':
               number_found = len([t for t in times_p_found])
               expected_number_found = N * mu * L * phi_pq_local[phenotype] * T
               ax.scatter([expected_number_found,], [number_found, ], s=3, alpha=0.7, c = ['b',])
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
      f.savefig('./plots/'+type_map+'mean_'+plot_type+'_plots'+string_saving_NC[:-1]+'.png', bbox_inches='tight', dpi=200)

