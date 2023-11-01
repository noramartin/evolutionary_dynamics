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

"""
################################################################################################################################################################################
## input: population properties
################################################################################################################################################################################
N = int(sys.argv[1])#, float(sys.argv[2])
phipq1, rho_p0 = float(sys.argv[2]), float(sys.argv[3])
s1_list = np.exp(np.linspace(np.log(4/N), np.log(0.6), num = 500))
K = 4


####################################################################################################
####################################################################################################
print('vary s1 - plot')
####################################################################################################
####################################################################################################
f,axarr=plt.subplots(ncols=2, figsize=(6, 2))#, gridspec_kw={'width_ratios': [1, 2]})
for i, (L, mu) in enumerate([(50, 0.000002), (500, 0.0000002)]):
    print('N', N, 'mu', mu, 'rho', rho_p0, 'phipq1', phipq1)
    string_saving = 'N'+g.format_integer_filename(N)+'_phipq1'+g.format_float_filename(phipq1) +'_rho_p0'+g.format_float_filename(rho_p0)+'_mu_long'+g.format_float_filename(mu)
    axarr[i].plot(s1_list, [const_rate_fixation_time(s1, N, phipq1, L, mu) for s1 in s1_list], 
                     c= 'k', label='theory\n(average-rate)', ls=':', lw=1) 
    axarr[i].plot(s1_list, [bursty_fixation_time(s1, N, phipq1, L, mu, rho_p0, K, polymorphic_correction=True) for s1 in s1_list], 
                     c= 'cyan', label='theory\n(random map)', ls=':', lw=1) 
    axarr[i].set_xscale('log')
    axarr[i].set_yscale('log')
    axarr[i].set_xlim(min(s1_list), max(s1_list))
    axarr[i].set_xlabel(r'selective advantage $s_1$')
    axarr[i].set_ylabel('mean fixation time\n(in generations)')
    #axarr.set_xlim(0.5 * min(s1_list), 2 * max(s1_list))
    if i == 1:
       axarr[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axarr[i].set_title('L='+str(L))
    axarr[i].annotate('ABCDEF'[i], xy=(0.06, 1.1), xycoords='axes fraction', fontsize=12, weight='bold')
f.tight_layout()
f.savefig('./plots/fixation_simple_theory'+string_saving+'_varys1_'+'.png', bbox_inches='tight', dpi=200)
del f, axarr
plt.close('all')"""
####################################################################################################
####################################################################################################
print('plot saturating function')
####################################################################################################
####################################################################################################
f,ax=plt.subplots(figsize=(3, 2))#, gridspec_kw={'width_ratios': [1, 2]})
x_list = np.power(10, np.linspace(-2, 3))
ax.plot(x_list, [1/(1+1/x) for x in x_list], c='k')
ax.set_xscale('log')
ax.set_xlabel(r'$M\ P^{fix}_{p_1}$')
ax.set_ylabel(r'$P^{fix}_{portal\ p_1}$')
ax.set_ylim(0, 1)
ax.set_xlim(min(x_list), max(x_list))
ax.set_yticks([0, 0.5, 1])
f.tight_layout()
f.savefig('./plots/saturating_function.png', bbox_inches='tight', dpi=200)
del f, ax
plt.close('all')

  
