import numpy as np
from .RNA_seq_structure_functions import sequence_int_to_str, get_dotbracket_from_int
import itertools  
from copy import deepcopy
import networkx as nx
from .neutral_component import get_NC_as_network
from networkx.algorithms import community

def neighbours_g_given_site(g, K, L, pos): 
   """list all pont mutational neighbours of sequence g (integer notation)"""
   return [tuple([oldK if gpos!=pos else new_K for gpos, oldK in enumerate(g)]) for new_K in range(K) if g[pos]!=new_K]


def find_average_number_neutral_mutations_each_site(NCindex, seq_vs_NCindex_array):
   K, L = seq_vs_NCindex_array.shape[1], seq_vs_NCindex_array.ndim
   site_vs_neutral_list = {s: [] for s in range(L)}
   site_vs_letters = {s: set([]) for s in range(L)}
   for g, NCindex_g in np.ndenumerate(seq_vs_NCindex_array):
      if NCindex == NCindex_g:
         for s in range(L):
            site_vs_neutral_list[s].append(len([seq_vs_NCindex_array[g2] for g2 in neighbours_g_given_site(g, K, L, s) if seq_vs_NCindex_array[g2] == NCindex]))
            site_vs_letters[s].add(sequence_int_to_str([g[s],]))
   for s, l in site_vs_neutral_list.items():
      if sum(l) == 0:
         assert len(site_vs_letters[s]) == 1
   return {s: np.mean(l) for s, l in site_vs_neutral_list.items() if sum(l) > 0}, site_vs_letters


def community_integer(letter_combinations, constrained_sites, g):
   letter_combination_given_seq = ''.join([sequence_int_to_str([g[s],]) for s in constrained_sites])
   return letter_combinations.index(letter_combination_given_seq)

def find_communities(NCindex, seq_vs_NCindex_array):
   K, L = seq_vs_NCindex_array.shape[1], seq_vs_NCindex_array.ndim
   average_number_neutral_mutations_each_site, site_vs_letters = find_average_number_neutral_mutations_each_site(NCindex, seq_vs_NCindex_array)
   NC_graph = get_NC_as_network(NCindex, seq_vs_NCindex_array)
   sites_by_constrainedness = sorted(average_number_neutral_mutations_each_site, key=average_number_neutral_mutations_each_site.get)
   number_sites_vs_modularity = {}
   number_sites_vs_partition = {}
   if len(sites_by_constrainedness) <= 2:
      print('very few partly constrained sites, NC size ', len(NC_graph.nodes()), 'NCindex', NCindex)
      return [set([deepcopy(n) for n in NC_graph.nodes()]),]
   for number_constrained_sites in range(1, len(sites_by_constrainedness) -1):
      constrained_sites = sorted(sites_by_constrainedness[:number_constrained_sites])
      letter_combinations = [''.join(x) for x in itertools.product(*[site_vs_letters[s] for s in constrained_sites])]
      community_dict = {c: [] for c in range(len(letter_combinations))}
      for g in NC_graph.nodes():
         community_dict[community_integer(letter_combinations, constrained_sites, g)].append(deepcopy(g))
      partition = [set(deepcopy(l)) for i, l in community_dict.items() if len(l) > 0]
      assert community.is_partition(NC_graph, partition)
      number_sites_vs_modularity[number_constrained_sites] = community.modularity(NC_graph, partition)
      del letter_combinations, community_dict
      number_sites_vs_partition[number_constrained_sites] = deepcopy(partition)
   best_number_constrained_sites = max(number_sites_vs_modularity.keys(), key=number_sites_vs_modularity.get)
   if len(number_sites_vs_modularity) > 1:
      print('highest modularity', number_sites_vs_modularity[best_number_constrained_sites], 'followed by', sorted([m for m in number_sites_vs_modularity.values()], reverse=True)[1])
   return number_sites_vs_partition[best_number_constrained_sites]

