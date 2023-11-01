import numpy as np
import networkx as nx
from .neutral_component import neighbours_g
from .RNA_seq_structure_functions import sequence_int_to_str, sequence_str_to_int


def NC_as_graph_for_plot(p1, p2, GPmap, V, relative_weight_nonneutral_edges=0.1):
   K, L = GPmap.shape[1], GPmap.ndim
   #find relevant p1 and p2 genotypes in mutational neighbourhood:
   G = nx.Graph()
   for gene in V:
      neighbours = neighbours_g(gene, K, L)
      G.add_node(sequence_int_to_str(gene), pheno='p0')
      for neighbourgene in neighbours:
         if GPmap[neighbourgene]==p1:
             G.add_node(sequence_int_to_str(neighbourgene), pheno='p1')
         if GPmap[neighbourgene]==p2:
             G.add_node(sequence_int_to_str(neighbourgene), pheno='p2')
   node_vs_pheno = nx.get_node_attributes(G, "pheno")
   for n1 in G.nodes():
      for n2_int in neighbours_g(sequence_str_to_int(n1), K, L):
         if sequence_int_to_str(n2_int) in G.nodes(): #hamming distance of 1
            edge_to_add = (n1, sequence_int_to_str(n2_int))
            if node_vs_pheno[edge_to_add[0]] == node_vs_pheno[edge_to_add[0]] == 'p0':
               G.add_edge(edge_to_add[0], edge_to_add[1], weight_for_drawing=1.0)
            else:
               G.add_edge(edge_to_add[0], edge_to_add[1], weight_for_drawing=relative_weight_nonneutral_edges)            
   return G

def has_p_neighbour(p, n, graph):
   node_vs_pheno = nx.get_node_attributes(graph, "pheno")
   for n2 in graph.neighbors(n):
      if node_vs_pheno[n2] == p:
         return True
   return False
