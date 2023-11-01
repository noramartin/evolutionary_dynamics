#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import NC_properties.general_functions as g
import pandas as pd
from matplotlib.lines import Line2D
import NC_properties.neutral_component as NC
import seaborn as sns
from NC_properties.RNA_seq_structure_functions import sequence_str_to_int, get_dotbracket_from_int
from collections import Counter
import phenotypes_to_draw as param


def calculate_coeff_variation(disc_list, min_number_discoveries=50, min_median=5):
   inter_disc_time_list = [disc_list[i] - disc_list[i-1] for i in range(1, len(disc_list)) if disc_list[i] - disc_list[i-1] > 0]
   ###
   if len(inter_disc_time_list) >= min_number_discoveries and np.median(inter_disc_time_list) >= min_median:
      mean_inter_disc_time=np.mean(inter_disc_time_list)
      std_inter_disc_time=np.std(inter_disc_time_list)
      return std_inter_disc_time/mean_inter_disc_time
   else:
      if len(inter_disc_time_list) < min_number_discoveries:
        print('NCindex, phenotype', NCindex, phenotype, 'insufficient appearances', len(inter_disc_time_list))
      else:
        print('NCindex, phenotype', NCindex, phenotype, 'median time too short', np.median(inter_disc_time_list))
      return np.nan



####################################################################################################
##input: GP map properties
####################################################################################################
K, L = 4, 12
min_number_discoveries = 500
min_median = 5
rare_pheno_min_f, rare_pheno_max_f = 0.001, 0.1
rare_pheno_max_f_for_plot = 0.05
set_other_NCs_in_neutral_set_to_zero = 1
if set_other_NCs_in_neutral_set_to_zero:
   string_NCs = 'otherNCszero'
else:
   string_NCs = ''
####################################################################################################
#### input: population parameters
####################################################################################################
N_LHS= 200# 500 # 200
mu_RHS= 0.00005 # 0.0002 # 0.00005
set_of_population_parametersLHS= [(N_LHS, max(g.roundT_up(10**3/(N_LHS * mu * rare_pheno_min_f * L)), 10**4), mu) for mu in (0.00005, 0.0001, 0.0005, 0.001, 0.005)] ##(0.00005, 0.0001, 0.0005, 0.001)
set_of_population_parametersRHS=[(N, max(g.roundT_up(10**3/(N * mu_RHS * rare_pheno_min_f * L)), 10**4), mu_RHS) for N in (10, 50, 100, 500)] ##(10, 20, 40, 80, 160, 320)
string_all_params = 'minf'+g.format_float_filename(rare_pheno_min_f)+'maxf'+g.format_float_filename(rare_pheno_max_f) + string_NCs
string_all_params +=  'LHS_N' + g.format_integer_filename(N_LHS) + '_mu_' + '_'.join([g.format_float_filename(p[2]) for p in set_of_population_parametersLHS])
string_all_params +=  'RHS_mu' + g.format_float_filename(mu_RHS) + '_N_' + '_'.join([g.format_integer_filename(p[0]) for p in set_of_population_parametersRHS])
string_all_params += '_min_median' + g.format_integer_filename(min_median) + 'min_number_discoveries' + g.format_integer_filename(min_number_discoveries)
####################################################################################################
####################################################################################################
plot_types = ['randommap', 'topology_map', 'community_map', 'RNA']
colors = {plot_types[3 -N]: color for N, color in enumerate(sns.color_palette('viridis', as_cmap=True)(np.linspace(0, 1.0, 4)))}
mapname_to_label = {'RNA': 'full RNA GP Map', 'randommap': 'random GP map', 'community_map': 'community GP map', 'topology_map': 'topology GP map'}
####################################################################################################
##input: GP map properties
####################################################################################################
df = pd.read_csv('./RNA_data/RNA_NCs_L'+str(L)+'_forrandommap.csv')
NCindex_list = df['NCindex'].tolist()
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
NCindex_vs_rho = {row['NCindex']: row['rho'] for i, row in df.iterrows()}
NCindex_vs_ph = {row['NCindex']: row['structure integer'] for i, row in df.iterrows()}
NCindex_vs_seq = {row['NCindex']: row['example sequence'] for i, row in df.iterrows()}
####################################################################################################
print('phenotypes with enough data for each NC')
####################################################################################################
NC_vs_pheno_vs_sufficient_data = {}
for NCindex in NCindex_list:
    p0 = NCindex_vs_ph[NCindex]
    rare_phenos = np.array([p for p, cpq in NC_vs_p_vs_cpq[NCindex].items() if p!=p0 and rare_pheno_min_f <= cpq <= rare_pheno_max_f_for_plot])
    ####
    NC_vs_pheno_vs_sufficient_data[NCindex] =  {p: True for p in rare_phenos}
    for phenoindex, phenotype in enumerate(rare_phenos):
        for plotindex, set_of_population_parameters in enumerate([set_of_population_parametersLHS, set_of_population_parametersRHS]):
            for parameterindex, (N, Tmax, mu) in enumerate(set_of_population_parameters):
            	for typemapindex, type_map in enumerate(plot_types):
                   if NC_vs_pheno_vs_sufficient_data[NCindex][phenotype]:
                       string_saving = 'neutral_N'+g.format_integer_filename(N)+'_T'+g.format_integer_filename(Tmax)+'_mu'+g.format_float_filename(mu)+'minf'+g.format_float_filename(rare_pheno_min_f)+'maxf'+g.format_float_filename(rare_pheno_max_f) +string_NCs+'NCindex'+str(NCindex) +'_'              
                       try:
                           times_p1_found=np.load('./evolution_data/'+type_map+'_times_p_found'+string_saving +str(phenotype)+'.npy')
                       except IOError:
                	       times_p1_found = []
                       if np.isnan(calculate_coeff_variation(times_p1_found, min_number_discoveries=min_number_discoveries, min_median = min_median)):
                           NC_vs_pheno_vs_sufficient_data[NCindex][phenotype] = False

####################################################################################################
print('burstiness plots only')
####################################################################################################
NCindex, phenotype = param.NCindex_first_plots, param.pr_first_plots
f,ax = plt.subplots(ncols=2, figsize=(8,3), dpi=60)
print('phenotype', phenotype)
for plotindex, set_of_population_parameters in enumerate([set_of_population_parametersLHS, set_of_population_parametersRHS]):
    x_list = [[mu * N * L for N, Tmax, mu in set_of_population_parameters], [N/L for N, Tmax, mu in set_of_population_parameters]][plotindex]
    maxy = 0
    for typemapindex, type_map in enumerate(plot_types):
        list_coeff_var = []
        for parameterindex, (N, Tmax, mu) in enumerate(set_of_population_parameters):
            string_saving = 'neutral_N'+g.format_integer_filename(N)+'_T'+g.format_integer_filename(Tmax)+'_mu'+g.format_float_filename(mu)+'minf'+g.format_float_filename(rare_pheno_min_f)+'maxf'+g.format_float_filename(rare_pheno_max_f) +string_NCs+'NCindex'+str(NCindex) +'_'              
            times_p1_found=np.load('./evolution_data/'+type_map+'_times_p_found'+string_saving +str(phenotype)+'.npy')
            coeff = calculate_coeff_variation(times_p1_found, min_number_discoveries=min_number_discoveries, min_median = min_median)
            list_coeff_var.append(coeff)
        ax[plotindex].plot(x_list, list_coeff_var, c=colors[type_map], marker='x')
        maxy = max(maxy, max(list_coeff_var))
        
    ax[plotindex].set_ylabel(r'coefficient of variation', size=12)
    ax[plotindex].set_xscale('log')
    ax[plotindex].set_xlabel([r'mutation supply $NLu$', r'population size/seq. length $N/L$'][plotindex], size=14) 
    ax[plotindex].set_ylim(0, maxy * 1.1)
    minx, maxy = min(x_list) * 0.8, max(x_list) * 1.2
    ax[plotindex].plot([minx, maxy], [1, 1], c='grey', ls=':')
    ax[plotindex].set_xlim(minx, maxy)
    
    ax[plotindex].annotate('ABCDEF'[plotindex], xy=(0.06, 1.1), xycoords='axes fraction', fontsize=16, weight='bold')
legend_content = [Line2D([0], [0], ms=10, marker='o', mew=0, lw=0.0, markerfacecolor=colors[label], label=mapname_to_label[label]) for label in plot_types]
leg = f.legend(handles = legend_content, fontsize=14, bbox_to_anchor=(0.5, 1.0001), loc='lower center', ncol=2)#, title_fontsize=16)
f.suptitle(r'initial structure: $p_0=$' + get_dotbracket_from_int(p0) + ', specifically NC containing sequence ' + NCindex_vs_seq[NCindex] +'\ntarget structure: ' + r'$p_1=$' + get_dotbracket_from_int(phenotype), y=0.1)
f.tight_layout()
f.savefig('./plots/singlerow'+ 'NCindex' + str(NCindex)+'phenotype'+str(phenotype)+'.png', bbox_inches='tight', dpi=200)
plt.close('all')
del f, ax
####################################################################################################
print('burstiness plots only - single figure for each NC')
####################################################################################################
from matplotlib.gridspec import SubplotSpec

def create_subtitle(fig, grid, title, fontsize):
    ### from https://stackoverflow.com/questions/27426668/row-titles-for-matplotlib-subplot
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold', fontsize=fontsize)
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')


for NCindex in NCindex_list:
    p0 = NCindex_vs_ph[NCindex]
    rare_phenos = np.array([p for p in NC_vs_pheno_vs_sufficient_data[NCindex].keys() if NC_vs_pheno_vs_sufficient_data[NCindex][p]])
    #if len(rare_phenos) < 3:
    #   continue
    nrows = int(np.ceil(len(rare_phenos)/2))
    f,ax = plt.subplots(ncols=5, nrows = nrows, figsize=(16,1.5 * len(rare_phenos)), dpi=60)
    grid = plt.GridSpec(nrows, 5, width_ratios=[1, 1, 0.25, 1, 1])
    for i in range(nrows):
        ax[i, 2].set_axis_off()
    if nrows * 2 > len(rare_phenos):
        ax[-1, 3].set_axis_off()
        ax[-1, 4].set_axis_off()
    ####################################################################################################
    print('plots for', NCindex)
    ####################################################################################################
    for phenoindex, phenotype in enumerate(rare_phenos):
        #print('phenotype', phenotype)
        if phenoindex >= nrows:
            create_subtitle(f, grid[phenoindex - nrows, 3:5], 'target structure: ' + r'$p_1=$' + get_dotbracket_from_int(phenotype), fontsize=12)
        else:
            create_subtitle(f, grid[phenoindex, 0:2], 'target structure: ' + r'$p_1=$' + get_dotbracket_from_int(phenotype), fontsize=12)
        for colindex, set_of_population_parameters in enumerate([set_of_population_parametersLHS, set_of_population_parametersRHS]):
            if phenoindex >= nrows:
               plotindex = (phenoindex- nrows, colindex + 3)
            else:
               plotindex = (phenoindex, colindex)
            x_list = [[mu * N * L for N, Tmax, mu in set_of_population_parameters], [N/L for N, Tmax, mu in set_of_population_parameters]][colindex]
            maxy = 0
            for typemapindex, type_map in enumerate(plot_types):
                list_coeff_var = []
                for parameterindex, (N, Tmax, mu) in enumerate(set_of_population_parameters):
                    string_saving = 'neutral_N'+g.format_integer_filename(N)+'_T'+g.format_integer_filename(Tmax)+'_mu'+g.format_float_filename(mu)+'minf'+g.format_float_filename(rare_pheno_min_f)+'maxf'+g.format_float_filename(rare_pheno_max_f) +string_NCs+'NCindex'+str(NCindex) +'_'              
                    times_p1_found=np.load('./evolution_data/'+type_map+'_times_p_found'+string_saving +str(phenotype)+'.npy')
                    coeff = calculate_coeff_variation(times_p1_found, min_number_discoveries=min_number_discoveries, min_median = min_median)
                    list_coeff_var.append(coeff)
                ax[plotindex].plot(x_list, list_coeff_var, c=colors[type_map], marker='x')
                maxy = max(maxy, max(list_coeff_var))
            ax[plotindex].set_ylabel(r'coefficient of variation', size=12)
            ax[plotindex].set_xscale('log')
            ax[plotindex].set_xlabel([r'mutation supply $NLu$', r'population size/seq. length $N/L$'][colindex], size=14) 
            ax[plotindex].set_ylim(0, maxy * 1.1)
            minx, maxy = min(x_list) * 0.8, max(x_list) * 1.2
            ax[plotindex].plot([minx, maxy], [1, 1], c='grey', ls=':')
            ax[plotindex].set_xlim(minx, maxy)
    legend_content = [Line2D([0], [0], ms=10, marker='o', mew=0, lw=0.0, markerfacecolor=colors[label], label=mapname_to_label[label]) for label in plot_types]
    leg = f.legend(handles = legend_content, fontsize=14, bbox_to_anchor=(0.5, 1.0001), loc='lower center', ncol=4)#, title_fontsize=16)
    #f.suptitle(r'initial structure: $p_0=$' + get_dotbracket_from_int(p0) + ', specifically NC containing sequence ' + NCindex_vs_seq[NCindex], y=0.1)
    print(NCindex, r'initial structure: $p_0=$' + get_dotbracket_from_int(p0) + ', specifically NC containing sequence ' + NCindex_vs_seq[NCindex])
    f.tight_layout()
    f.savefig('./plots/singlerow'+ 'NCindex' + str(NCindex)+'.png', bbox_inches='tight', dpi=200)
    plt.close('all')
    del f, ax
