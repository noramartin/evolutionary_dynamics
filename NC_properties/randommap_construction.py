import numpy as np
from NC_properties.RNA_seq_structure_functions import *
import NC_properties.neutral_component as NC
from copy import deepcopy

def create_randommap(GPmapRNA, NCindex_array, NCindex, phi_pq_local):
   phenotype = GPmapRNA[NC.find_index(NCindex, NCindex_array)]
   assert abs(sum([phi for phi in phi_pq_local.values()]) - 1) < 0.01
   L, K = GPmapRNA.ndim, GPmapRNA.shape[0]
   #phi_pq_local = phipq.phi_pq_RNA_NC(GPmapRNA, NCindex_array, NCindex, filename='./RNA_data/RNA_phipqNC_L'+str(L)+'NCindex'+str(NCindex)+'.npy')
   assert phenotype in phi_pq_local
   local_neutral_set_sizes = {int(ph): int(round(phipq * K**L)) for ph, phipq in phi_pq_local.items()}
   assert abs(sum(local_neutral_set_sizes.values())-K**L) <= len(phi_pq_local) + 1
   ## correct rounding errors
   while abs(sum(local_neutral_set_sizes.values()) - K**L) > 0:
      index = np.random.choice(list(local_neutral_set_sizes.keys()))
      if sum(local_neutral_set_sizes.values()) > K**L and local_neutral_set_sizes[index] > 0:
         local_neutral_set_sizes[index] -= 1
      elif sum(local_neutral_set_sizes.values()) < K**L and local_neutral_set_sizes[index] > 0:
         local_neutral_set_sizes[index] += 1
   assert sum(local_neutral_set_sizes.values()) == K**L
   ## construct map
   phenotypes_ordered = [ph for ph, N in local_neutral_set_sizes.items() for i in range(N) if N > 0] #arrange phenotypes in line
   assert len(phenotypes_ordered) == K**L
   phenotypes_list = np.random.permutation(phenotypes_ordered) #permute
   GPmap_randommap = np.reshape(phenotypes_list, (K,)*L) #bring into shape of genotype space
   del phenotypes_list, phenotypes_ordered, local_neutral_set_sizes
   ## test
   structure_integers, structure_counts = np.unique(GPmap_randommap, return_counts=True)
   for struct_count, structure in zip(structure_counts, structure_integers):
      assert  abs(struct_count - int(round(phi_pq_local[structure] * K**L))) < 5
   return GPmap_randommap

def create_topology_preserving_randommap_second_method(GPmapRNA, NCindex_array, NCindex, phi_pq_local):
   phenotype = GPmapRNA[NC.find_index(NCindex, NCindex_array)]
   L, K = GPmapRNA.ndim, GPmapRNA.shape[0]
   GPmap_shuffled = GPmapRNA.copy()
   ## extract position vs neighbour for all neighbours
   NCsize = len([NCindex for genotype, NCindex2 in np.ndenumerate(NCindex_array) if NCindex2 == NCindex])
   print('NCsize', NCsize)
   norm = sum([phipq for ph, phipq in phi_pq_local.items() if ph != phenotype])
   local_neutral_set_sizes = {int(ph): int(round(phipq/norm * (K**L - NCsize))) for ph, phipq in phi_pq_local.items() if ph != phenotype}
   assert abs(sum(local_neutral_set_sizes.values()) + NCsize - K**L) <= len(phi_pq_local) 
   ## construct map
   phenotypes_ordered = [ph for ph, N in local_neutral_set_sizes.items() for i in range(N) if N > 0] #arrange phenotypes in line
   assert abs(len(phenotypes_ordered) + NCsize - K**L) <= len(phi_pq_local)    
   phenotypes_list = np.random.permutation(phenotypes_ordered) #permute
   next_phenotype_index = 0
   for genotype, NCindex2 in np.ndenumerate(NCindex_array):
      if NCindex2 != NCindex:
         GPmap_shuffled[genotype] = phenotypes_list[next_phenotype_index] #np.random.choice(ph_list, p=phi_list_norm)
         next_phenotype_index += 1
         if next_phenotype_index >= len(phenotypes_list):
            next_phenotype_index = 0
   return GPmap_shuffled



def create_topology_preserving_randommap_with_community_neighbourhood_intact(GPmapRNA, genotype_vs_partition):
   ### for each community, we need a dictionary of genotype in neighbourhood versus phenotype - here neighbour genotypes should only be included if they only belong to one community
   community_vs_neighbour_vs_pheno = {}
   neighbour_vs_communities = {}
   neighbour_vs_number_connections = {}
   phenotype = GPmapRNA[tuple([g for g in genotype_vs_partition.keys()][0])]
   L, K = GPmapRNA.ndim, GPmapRNA.shape[0]
   for genotype, partition in genotype_vs_partition.items():
      for n in NC.neighbours_g(genotype, K, L):
         n_pheno = GPmapRNA[tuple(n)]
         if n_pheno != phenotype:
            try:
               community_vs_neighbour_vs_pheno[partition][tuple([x for x in n])] = n_pheno 
            except KeyError:
               community_vs_neighbour_vs_pheno[partition] = {tuple([x for x in n]): n_pheno}
            try:
               neighbour_vs_communities[tuple([x for x in n])].append(partition)
            except KeyError:
               neighbour_vs_communities[tuple([x for x in n])] = [partition,]
            try:
               neighbour_vs_number_connections[tuple([x for x in n])] += 1
            except KeyError:
               neighbour_vs_number_connections[tuple([x for x in n])] = 1
   # delete neighbours that are neighbours of two communities
   for n, partitions in neighbour_vs_communities.items():
      if len(partitions) > 1:
         for partition in partitions:
            if tuple([x for x in n]) in community_vs_neighbour_vs_pheno[partition].keys():
               del community_vs_neighbour_vs_pheno[partition][tuple([x for x in n])]
   #swap as before, but for each community
   GPmap_shuffled = GPmapRNA.copy()
   for partition in community_vs_neighbour_vs_pheno:
      neighbour_number_mutations = sorted(list(set([m for n, m in neighbour_vs_number_connections.items() if n in community_vs_neighbour_vs_pheno[partition]])))
      for m in neighbour_number_mutations:
         neighbour_locations, neighbours_identities_shuffled_tuple = deepcopy(zip(*[(g, p) for g, p in community_vs_neighbour_vs_pheno[partition].items() if neighbour_vs_number_connections[g] == m]))
         neighbours_identities_shuffled = [deepcopy(n) for n in neighbours_identities_shuffled_tuple]
         np.random.shuffle(neighbours_identities_shuffled)
         new_neighbour_vs_pheno = {l: i for l, i in zip(neighbour_locations, neighbours_identities_shuffled)}
         for new_geno, new_pheno in new_neighbour_vs_pheno.items():
            GPmap_shuffled[new_geno] = new_pheno
   return GPmap_shuffled
     


