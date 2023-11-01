import numpy as np
import networkx as nx
from .RNA_seq_structure_functions import isundefinedpheno, get_dotbracket_from_int
from os.path import isfile
from functools import partial
from multiprocessing import Pool
import networkx as nx
import os
#import memory_profiler as mem_profile
#from guppy import hpy
from collections import Counter

base_to_number={'A':0, 'C':1, 'U':2, 'G':3, 'T':2}



def find_index(N, array):
   """find index of first occurence of N in array (within 10**-6)"""
   index = np.unravel_index(np.argmax(np.abs(array-N) < 10.0**-6, axis=None), array.shape)
   assert abs(array[tuple(index)]-int(N)) < 10.0**(-6)
   return index


############################################################################################################
## point-mutational neighbours
############################################################################################################
def neighbours_g(g, K, L): 
   """list all pont mutational neighbours of sequence g (integer notation)"""
   return [tuple([oldK if gpos!=pos else new_K for gpos, oldK in enumerate(g)]) for pos in range(L) for new_K in range(K) if g[pos]!=new_K]

############################################################################################################
## identify neutral components
############################################################################################################
def find_all_NCs_parallel(GPmap, filename=None, isundefinedphenofunction = isundefinedpheno):
   """find individual neutral components for all non-unfolded structures in GP map;
   saves/retrieves information if filename given and files present; otherwise calculation from scratch"""
   K, L = GPmap.shape[1], GPmap.ndim
   if not filename or not isfile(filename):
      structure_list = [ph for ph in np.unique(GPmap) if not isundefinedphenofunction(ph)]
      find_NC_correct_const_arguments = partial(find_NC, GPmap=GPmap)
      #with Pool(20) as pool:
      #   NC_per_structure_list = pool.map(find_NC_correct_const_arguments, structure_list) 
      NC_per_structure_list = [find_NC_correct_const_arguments(s) for s in structure_list]
      NCindex_global_count = 1 #such that 0 remains for undefined structure
      seq_vs_NCindex_array = np.zeros((K,)*L, dtype='uint32')
      for structureindex, structure in enumerate(structure_list):
         NC_dict_this_structure = NC_per_structure_list[structureindex]
         map_to_global_NCindex = {NCindex: NCindex_global_count + NCindex for NCindex in set(NC_dict_this_structure.values())}
         NCindex_global_count = max(map_to_global_NCindex.values()) + 1
         for seq, NCindex in NC_dict_this_structure.items():
            seq_vs_NCindex_array[seq] = map_to_global_NCindex[NCindex]
      if filename:
         np.save(filename, seq_vs_NCindex_array)
   else:
      seq_vs_NCindex_array = np.load(filename)
   return seq_vs_NCindex_array

def find_NC(structure_int, GPmap):
   """find individual neutral components in the neutral set of the structure structure_int in GPmap;
   NCindex will start at 0"""
   print('find NC of structure', structure_int, flush=True)
   K, L = GPmap.shape[1], GPmap.ndim
   neutral_set = nx.Graph()
   neutral_set.add_nodes_from([tuple(g) for g in np.argwhere(GPmap==structure_int)])
   for g in neutral_set.nodes():
      for g2 in neighbours_g(g, K, L):
         if GPmap[g2] == structure_int:
            neutral_set.add_edge(g, g2)
   geno_vs_NC = {g: NCindex for NCindex, list_of_geno in enumerate(nx.connected_components(neutral_set)) for g in list_of_geno}
   return geno_vs_NC


############################################################################################################
## NC as network
############################################################################################################
def get_NC_as_network(NCindex, seq_vs_NCindex_array):
   """convert the NC with index NCindex to a networkx network (for plotting)"""
   K, L =  seq_vs_NCindex_array.shape[1], seq_vs_NCindex_array.ndim
   neutral_component_list_of_seq = [tuple([x for x in g]) for g in np.argwhere(seq_vs_NCindex_array==NCindex)]
   NC_graph = nx.Graph()
   NC_graph.add_nodes_from(neutral_component_list_of_seq)
   for g1 in neutral_component_list_of_seq:
      for g2 in neighbours_g(g1, K, L):
         if seq_vs_NCindex_array[g2] == NCindex:
            NC_graph.add_edge(g1, g2)   
   return NC_graph

############################################################################################################
## phi pq between neutral sets
############################################################################################################
#@profile
def phi_pq_GPmap(GPmap, ph_vs_size):
   K, L =  GPmap.shape[1], GPmap.ndim
   q_vs_p_vs_phipq = {q: {p: 0 for p in ph_vs_size} for q in ph_vs_size}
   count = 0
   for g1, ph1 in np.ndenumerate(GPmap):
      count += 1
      if not isundefinedpheno(ph1):
         neighbour_phenos = Counter([GPmap[g2] for g2 in neighbours_g(g1, K, L) if not isundefinedpheno(GPmap[g2])])
         for ph2, c in neighbour_phenos.items():
            q_vs_p_vs_phipq[ph1][ph2] += 1.0* c/(L * (K-1) * ph_vs_size[ph1])
         del neighbour_phenos
         if count % 10**5 == 0:
            print('finished', count*100.0/4**L, '%')
   print('finished function')
   return q_vs_p_vs_phipq
############################################################################################################
## phi pq between NCs
############################################################################################################
#@profile
def NC_vs_cqp_info(GPmap, seq_vs_NCindex_array, ph_list):
   NC_indices_withundefined, NC_sizes_withundefined = np.unique(seq_vs_NCindex_array, return_counts=True)
   NCindex_vs_size = {NCindex: size for NCindex, size in zip(NC_indices_withundefined, NC_sizes_withundefined) if NCindex > 0}
   K, L =  seq_vs_NCindex_array.shape[1], seq_vs_NCindex_array.ndim
   NC_vs_p_vs_cpq = {NCindex: {p: 0 for p in ph_list} for NCindex in NCindex_vs_size}
   NC_vs_p_vs_fraction_portal_genotypes = {NCindex: {p: 0 for p in ph_list} for NCindex in NCindex_vs_size}
   count = 0
   for g1, NCindex in np.ndenumerate(seq_vs_NCindex_array):
      count += 1
      if NCindex > 0:
         assert NCindex == seq_vs_NCindex_array[g1]
         neighbour_phenos = Counter([GPmap[g2] for g2 in neighbours_g(g1, K, L)])
         for ph, c in neighbour_phenos.items():
            NC_vs_p_vs_cpq[NCindex][ph] += 1.0* c/(L * (K-1) * NCindex_vs_size[NCindex])
            NC_vs_p_vs_fraction_portal_genotypes[NCindex][ph] += 1.0/NCindex_vs_size[NCindex]
         del neighbour_phenos
         if count % 10**5 == 0:
            print('finished', count*100.0/K**L, '%')
   print('finished function')
   return NC_vs_p_vs_cpq, NC_vs_p_vs_fraction_portal_genotypes


def NC_vs_rho(GPmap, seq_vs_NCindex_array):
   NC_indices_withundefined, NC_sizes_withundefined = np.unique(seq_vs_NCindex_array, return_counts=True)
   NCindex_vs_size = {NCindex: size for NCindex, size in zip(NC_indices_withundefined, NC_sizes_withundefined) if NCindex > 0}
   K, L =  seq_vs_NCindex_array.shape[1], seq_vs_NCindex_array.ndim
   NC_vs_rho = {NCindex: 0 for NCindex in NCindex_vs_size.keys()}
   for g1, NCindex in np.ndenumerate(seq_vs_NCindex_array):
      if NCindex > 0:
         assert NCindex == seq_vs_NCindex_array[g1]
         neighbour_neutral_phenos = len([GPmap[g2] for g2 in neighbours_g(g1, K, L) if NCindex == seq_vs_NCindex_array[g2]])
         NC_vs_rho[NCindex] += 1.0* neighbour_neutral_phenos/(L * (K-1) * NCindex_vs_size[NCindex])
   return NC_vs_rho

############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":

   ##########################################
   print('test mutational neighbours')
   ##########################################
   def Hamming_dist(g1, g2):
      return len([1 for x, y in zip(g1, g2) if x != y])

   g, K, L = (1, 2, 3, 1), 4, 4
   point_mutational_neighbours = neighbours_g(g, K, L)
   point_mutational_neighbours_reference = [(0, 2, 3, 1), (2, 2, 3, 1), (3, 2, 3, 1), (1, 0, 3, 1), (1, 1, 3, 1), (1, 3, 3, 1),
                                            (1, 2, 0, 1), (1, 2, 1, 1), (1, 2, 2, 1), (1, 2, 3, 0), (1, 2, 3, 2), (1, 2, 3, 3)]
   assert len(point_mutational_neighbours) == len(point_mutational_neighbours_reference)
   for mutational_neighbour in point_mutational_neighbours_reference:
      assert mutational_neighbour in point_mutational_neighbours
   for test_no in range(1000):
      seq = tuple(np.random.choice(np.arange(4), L, replace=True)) 
      point_mutational_neighbours = neighbours_g(seq, K, L)
      assert len(point_mutational_neighbours) == (K-1) * L == len(set(point_mutational_neighbours))
      for seq2 in point_mutational_neighbours:
         assert Hamming_dist(seq2, seq) == 1
   ##########################################
   print('neutral component: toy examples')
   ##########################################
   def create_epistasis_testcase(seq_vs_struct, K=4):
      """ build test case arrays to test NC detection"""
      L = len(list(seq_vs_struct.keys())[0])
      testGPmap = np.zeros((K,)*L, dtype='uint32')
      for seq, structure in seq_vs_struct.items():
         testGPmap[seq] = structure
      return testGPmap
   seq_vs_struct = {(0, 0, 0): 50, (0, 1, 0): 50, (2, 0, 0): 50, (1, 1, 1): 50, (1, 2, 1): 50, (2, 0, 1): 82, (0, 0, 1): 82, (0, 1, 1): 82}
   testGPmap = create_epistasis_testcase(seq_vs_struct, K=4)
   seq_vs_NCindex_array  = find_all_NCs_parallel(testGPmap)
   NC_indices_withundefined, NC_sizes_withundefined = np.unique(seq_vs_NCindex_array, return_counts=True)
   NCindex_vs_NCsize = {NCindex: NCsize for NCindex, NCsize in zip(NC_indices_withundefined, NC_sizes_withundefined)}
   for seq, NCindex in np.ndenumerate(seq_vs_NCindex_array):
      assert NCindex == 0 or seq in seq_vs_struct
   assert seq_vs_NCindex_array[(0, 0, 0)] == seq_vs_NCindex_array[(0, 1, 0)] == seq_vs_NCindex_array[(2, 0, 0)] != seq_vs_NCindex_array[(1, 1, 1)]
   assert seq_vs_NCindex_array[(1, 1, 1)] == seq_vs_NCindex_array[(1, 2, 1)] != seq_vs_NCindex_array[(0, 0, 1)]
   assert seq_vs_NCindex_array[(2, 0, 1)] == seq_vs_NCindex_array[(0, 0, 1)] == seq_vs_NCindex_array[(0, 1, 1)] != seq_vs_NCindex_array[(0, 0, 0)]
   assert NCindex_vs_NCsize[seq_vs_NCindex_array[(0, 0, 0)]] == 3
   assert NCindex_vs_NCsize[seq_vs_NCindex_array[(1, 1, 1)]] == 2
   assert NCindex_vs_NCsize[seq_vs_NCindex_array[(2, 0, 1)]] == 3
   assert testGPmap[find_index(seq_vs_NCindex_array[(0, 0, 0)], seq_vs_NCindex_array)] == 50




