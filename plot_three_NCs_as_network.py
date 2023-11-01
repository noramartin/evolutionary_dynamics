#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
from NC_properties.RNA_seq_structure_functions import sequence_str_to_int, get_dotbracket_from_int
import RNA
from NC_properties.NC_network_for_plotting import *
from NC_properties.neutral_component import neighbours_g
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from evolution_functions.evolutionary_dynamics_function_fixation_numba import find_startgeno_randommap
import phenotypes_to_draw as param

def analyse_NC(parameters_NC, GPmapRNA, NCindex_array, K, L, print_characteristics=False, fixed_nodes = [], fixed_pos = {}):
    startgene_str, p0, p1, p2 = parameters_NC[0], int(parameters_NC[1]), int(parameters_NC[2]), int(parameters_NC[3])
    NCindex = NCindex_array[sequence_str_to_int(startgene_str)]
    string_saving = 'p1_'+str(p1)+'p2_'+str(p2) + startgene_str + 'L'+str(L)
    assert GPmapRNA[sequence_str_to_int(startgene_str)] == p0
    ##############################################################################################################################
    print('collect data on NC', flush=True)
    ##############################################################################################################################
    V = np.argwhere(NCindex_array==NCindex)
    G = NC_as_graph_for_plot(p1, p2, GPmapRNA, V, relative_weight_nonneutral_edges=0.4)
    node_vs_pheno = nx.get_node_attributes(G, "pheno")
    
    ##############################################################################################################################
    print('print a few key characteristics', flush=True)
    ##############################################################################################################################
    numberseq_p1accessible = len([n for n, ndata in G.nodes(data=True) if ndata['pheno'] == 'p0' and has_p_neighbour('p1', n, G)])
    numberseq_p2accessible = len([n for n, ndata in G.nodes(data=True) if ndata['pheno'] == 'p0' and has_p_neighbour('p2', n, G)])
    numberseq_both_accessible = len([n for n, ndata in G.nodes(data=True) if ndata['pheno'] == 'p0' and has_p_neighbour('p1', n, G) and has_p_neighbour('p2', n, G)])
    if print_characteristics:
       print('expected cpq', NC_vs_p_vs_cpq[NCindex][p1], NC_vs_p_vs_cpq[NCindex][p2], flush=True)
       print('NC size of p0: ', len([n for n, ndata in G.nodes(data=True) if ndata['pheno'] == 'p0']), flush=True)
       print('number p1 shown: ', len([n for n, ndata in G.nodes(data=True) if ndata['pheno'] == 'p1']), flush=True)
       print('number p2 shown: ', len([n for n, ndata in G.nodes(data=True) if ndata['pheno'] == 'p2']), flush=True)
       print('number of genotypes from which p1 accessible: ', numberseq_p1accessible, flush=True)
       print('number of genotypes from which p2 accessible: ', numberseq_p2accessible, flush=True)
       print('number of genotypes from which both p1 and p2 accessible: ', numberseq_both_accessible)
       print('mean distance between p2 and nearest p1:',np.mean([min([nx.shortest_path_length(G, source=n1, target=n2) for n1, n1data in G.nodes(data=True) if n1data['pheno']=='p1']) for n2, n2data in G.nodes(data=True) if n2data['pheno']=='p2']))
       print('bias in number of p0 from which p1/p2 accessible:', numberseq_p1accessible/float(numberseq_p2accessible), flush=True)
       print('bias: '+str(NC_vs_p_vs_cpq[NCindex][p1]/NC_vs_p_vs_cpq[NCindex][p2]), flush=True)
    #nx.write_gpickle(G, './RNA_data/graphNC_RNA12_g'+string_saving+'.gpickle')         
    ##############################################################################################################################
    print('node positions', flush=True)
    ##############################################################################################################################
    #pheno_to_color={'p0': 'r', 'p1': 'b', 'p2': 'g'}
    print('number of nodes', len(node_vs_pheno), NCindex)
    if len(node_vs_pheno) < 7000:
    	iterations = 100
    else:
    	iterations = 40
    if len(fixed_nodes):
       pos=nx.spring_layout(G, iterations=iterations, weight='weight_for_drawing', pos=fixed_pos, fixed=fixed_nodes)
    else:
    	 pos=nx.spring_layout(G, iterations=iterations, weight='weight_for_drawing')
    print('finished', flush=True)
    return deepcopy(pos), G.copy(), string_saving


def network_breadth_first(p0, GPmap, K, L, g0, recursions_left = 2, random_walk_genes=[]):
  neighbourhood_list = [tuple(g) for g in neighbours_g(g0, K, L)]
  random_walk_genes += [deepcopy(g) for g in neighbourhood_list if GPmap[g] == p0] + [deepcopy(g0),]
  if recursions_left > 0:
     for g in neighbourhood_list:
        if GPmap[g] == p0:
           random_walk_genes += network_breadth_first(p0, GPmap, K, L, recursions_left=recursions_left-1, g0=deepcopy(g), random_walk_genes=random_walk_genes)
  random_walk_genes = list(set(random_walk_genes))
  assert len(random_walk_genes) == len([1 for g in random_walk_genes if GPmap[g] == p0])
  return random_walk_genes

	
def analyse_excerpt_randommap(parameters_NC, GPmapRNA, K, L, size=1000):
  p0, p1, p2 = int(parameters_NC[0]), int(parameters_NC[1]), int(parameters_NC[2])
  string_saving = 'p1_'+str(p1)+'p2_'+str(p2) + startgene_str + 'L'+str(L)
  ##############################################################################################################################
  print('collect data on NC', flush=True)
  ##############################################################################################################################
  V = network_breadth_first(p0, GPmapRNA, K, L, recursions_left = 2, random_walk_genes = [], g0= tuple(find_startgeno_randommap(GPmapRNA, p0)))
  G = NC_as_graph_for_plot(p1, p2, GPmapRNA, V, relative_weight_nonneutral_edges=0.5)
  node_vs_pheno = nx.get_node_attributes(G, "pheno")
  print('expected cpq', NC_vs_p_vs_cpq[NCindex][p1], NC_vs_p_vs_cpq[NCindex][p2], flush=True)    
  ##############################################################################################################################
  print('node positions', flush=True)
  ##############################################################################################################################
  #pheno_to_color={'p0': 'r', 'p1': 'b', 'p2': 'g'}
  print('number of nodes', len(node_vs_pheno), NCindex)
  if len(node_vs_pheno) < 7000:
    iterations = 100
  else:
    iterations = 40
  pos=nx.spring_layout(G, iterations=iterations, weight='weight_for_drawing')
  print('finished', flush=True)
  return deepcopy(pos), G.copy(), string_saving

##############################################################################################################################
print('load data: GP map')
##############################################################################################################################
K, L = 4, 12
GPmapRNA = np.load('./RNA_data/RNA_GPmap_L'+str(L)+'.npy') #gp map
NCindex_array = np.load('./RNA_data/RNA_NCindexArray_L'+str(L)+'.npy')
##############################################################################################################################
print('load cpq data')
##############################################################################################################################
filename_cpq = './RNA_data/RNA_NCs_cqp_data_L'+str(L)+'.csv'
df_cpq, NC_vs_p_vs_cpq = pd.read_csv(filename_cpq), {}
for i, row in df_cpq.iterrows():
    try:
       NC_vs_p_vs_cpq[int(row['NCindex'])][int(row['new structure integer'])] = row['cpq']
    except KeyError:
       NC_vs_p_vs_cpq[int(row['NCindex'])] = {int(row['new structure integer']): row['cpq']}


##############################################################################################################################
print('parameters: p0, p1, p2')
##############################################################################################################################
parameters_list = [(param.seq_firstplots, param.pg_first_plots, param.pb_first_plots, param.pr_first_plots, 'firstplots'),
                   (param.seq_twopeaked, param.p0_twopeaked, param.p1_twopeaked, param.p2_twopeaked, 'twopeaked')]

##############################################################################################################################
print('plot')
##############################################################################################################################
for startgene_str, p0, p1, p2, str_save in parameters_list:
    NCindex = NCindex_array[sequence_str_to_int(startgene_str)]
    for mapindex, type_map in enumerate(['randommap', 'RNA', 'topology_map', 'community_map']):
       if type_map == 'RNA':
           GPmapRNA = np.load('./'+type_map+'_data/'+type_map+'_GPmap_L'+str(L)+'.npy')
           for i, p in enumerate([p0, p1, p2]):
              RNA.PS_rna_plot(' '*L, get_dotbracket_from_int(p), './plots_structures/'+str_save+'p'+str(i)+'_'+str(p)+'.ps')
           print_characteristics = True
       else:
           try:
              if type_map == 'randommap':
                 GPmapRNA = np.load('./'+type_map+'_data/'+type_map+'GPmap_L'+str(L)+'NCindex'+str(NCindex)+'.npy') 
              else:
                 GPmapRNA = np.load('./'+type_map+'_data/'+type_map+'_GPmap_L'+str(L)+'NCindex'+str(NCindex)+'.npy') 
              print_characteristics = False
           except IOError:
             print('not found', './'+type_map+'_data/'+type_map+'_GPmap_L'+str(L)+'NCindex'+str(NCindex)+'.npy')
             continue
       if mapindex > 1: #use existing positions
          pos, G, string_saving = analyse_NC((startgene_str, p0, p1, p2), GPmapRNA=GPmapRNA, NCindex_array=NCindex_array, K=K, L=L, print_characteristics=print_characteristics, fixed_nodes = nodelist0_fixed, fixed_pos=pos_list_fixed)
       elif mapindex == 1: #define topology positions
           pos, G, string_saving = analyse_NC((startgene_str, p0, p1, p2), GPmapRNA=GPmapRNA, NCindex_array=NCindex_array, K=K, L=L, print_characteristics=print_characteristics)          	  
       else:
           pos, G, string_saving = analyse_excerpt_randommap((p0, p1, p2), GPmapRNA=GPmapRNA, K=K, L=L, size=10**3)
       nodelist0 = [n for n, data in G.nodes(data=True) if data['pheno'] == 'p0']
       nodelist1 = [n for n, data in G.nodes(data=True) if data['pheno'] == 'p1']
       nodelist2 = [n for n, data in G.nodes(data=True) if data['pheno'] == 'p2']
       nx.draw_networkx_nodes(G, pos, node_color='grey', node_size=2.5, linewidths=0.0, alpha=0.8, nodelist = nodelist0)#, zorder=-2)
       nx.draw_networkx_nodes(G, pos, node_color='b', node_size=5, linewidths=0.0, alpha=0.99, nodelist = nodelist1)#, zorder=-2)
       nx.draw_networkx_nodes(G, pos, node_color='r', node_size=5, linewidths=0.0, alpha=0.99, nodelist = nodelist2,)# zorder=-2)
       nx.draw_networkx_edges(G,pos,width=0.12,alpha=0.55, edge_color='k')
       plt.axis('off')
       plt.savefig('./plots_arrival_frequent/'+str_save+'network'+ 'NCindex'+str(NCindex)+string_saving + type_map+'.png', bbox_inches='tight', transparent=True, dpi=1000)
       plt.close('all')
       ### save positions
       nodelist0_fixed = deepcopy(nodelist0)
       pos_list_fixed = deepcopy(pos)

##############################################################################################################################
print('plot networks with no connections')
##############################################################################################################################
df = pd.read_csv('./RNA_data/RNA_NCs_L'+str(L)+'_forrandommap.csv')
NCindex_list = df['NCindex'].tolist()
structure_int_list = df['structure integer'].tolist()
startgene_list = df['example sequence'].tolist()
GPmapRNA = np.load('./RNA_data/RNA_GPmap_L'+str(L)+'.npy') 
f, ax = plt.subplots(ncols = len(NCindex_list), figsize=(4* len(NCindex_list), 4))
for i, NCindex in enumerate(NCindex_list):
    pos, G, string_saving = analyse_NC((startgene_list[i], structure_int_list[i], -2, -2), GPmapRNA=GPmapRNA, NCindex_array=NCindex_array, K=K, L=L, print_characteristics=False)   
    nodelist0 = [n for n, data in G.nodes(data=True) if data['pheno'] == 'p0']        
    nx.draw_networkx_nodes(G, pos, node_color='grey', node_size=2.5, linewidths=0.0, alpha=0.8, nodelist = nodelist0, ax=ax[i])#, zorder=-2)
    nx.draw_networkx_edges(G,pos,width=0.12,alpha=0.55, edge_color='k', ax=ax[i])
    ax[i].axis('off')
    ax[i].set_title('ABCDEF'[i]+') '+r'initial structure:'+'\n'+'$p_0=$' + get_dotbracket_from_int(structure_int_list[i]) + '\nsequence'+'\n' + startgene_list[i], fontsize=12)
f.savefig('./plots/network'+ 'NCindices_forrandommap_NCs_'+'_'.join([str(NCindex) for NCindex in NCindex_list])+'.png', bbox_inches='tight', transparent=True, dpi=1000)
plt.close('all')
del f, ax






