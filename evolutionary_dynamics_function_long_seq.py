#!/usr/bin/env python
import numpy as np
from numba import jit, prange
from collections import Counter
from NC_properties.RNA_seq_structure_functions import get_mfe_structure_as_int
from copy import deepcopy

@jit(nopython=True)
def select_next_generation_for_numba(N, P_list):
   cumsum_P = np.cumsum(P_list)
   selected_index = np.zeros(N, dtype=np.uint32)
   for i in range(N):
      randomnumber = np.random.rand()
      selected_ind = N * 2 # if this was ever chosen because the loop did not stop, this would raise error
      for ind in range(N):
         if ind > 0 and cumsum_P[ind-1] <= randomnumber < cumsum_P[ind]:
            assert selected_ind == 2*N # check that only one individual meets condition
            assert -0.01 < cumsum_P[ind] - cumsum_P[ind-1] -  P_list[ind] < 0.01 # check cumsum
            selected_ind = ind
         elif ind == 0 and 0 <= randomnumber < cumsum_P[ind]:
            assert selected_ind == 2*N # check that only one individual meets condition
            assert -0.01 < cumsum_P[ind] -  P_list[ind] < 0.01 # check cumsum
            selected_ind = ind
         elif ind == N-1 and cumsum_P[ind-1] <= randomnumber:
            assert selected_ind == 2*N # check that only one individual meets condition
            assert -0.01 < cumsum_P[ind] - 1 < 0.01 # next one sums to one
            assert -0.01 < 1 - cumsum_P[ind-1] -  P_list[ind] < 0.01 # check cumsum
            selected_ind = ind
      selected_index[i] = selected_ind
      #if selected_ind > N:
      #   print(randomnumber)
   assert np.max(selected_index) < N
   return selected_index


@jit(nopython=True)
def select_mutation_for_numba(mu):
   randomnumber = np.random.rand()
   if randomnumber > mu:
      return False
   else:
      return True

@jit(nopython=True)
def fitness_singlepeak(ph, p0):
   if ph==p0:
      return 1
   else:
      return 0

@jit(nopython=True)
def get_new_generation(P_fitness_current, P_g_current, P_g_new, N, mu, K, L ):
   #choose next generation
   P_list = np.divide(P_fitness_current[:],float(np.sum(P_fitness_current[:])))
   assert len(P_list) == N
   rows=select_next_generation_for_numba(N, P_list=P_list) #select N rows that get chosen for the next generation
   for individual in range(N):
      P_g_new[individual,:] = P_g_current[rows[individual],:] #fill in genes for next generation
      #mutate with p=mu
      for pos in range(L):
         mutate=select_mutation_for_numba(mu)
         if mutate:
            alternative_bases = np.array([b for b in range(K) if b!=P_g_new[individual,pos]])
            P_g_new[individual,pos] = np.random.choice(alternative_bases)
   return P_g_new

@jit(nopython=True)
def neighbours_g_nested_list(g, K, L): 
   """list all pont mutational neighbours of sequence g (integer notation)"""
   return [[oldK if gpos!=pos else new_K for gpos, oldK in enumerate(g)] for pos in range(L) for new_K in range(K) if g[pos]!=new_K]


def update_GP_dict(current_GP_dict, P_g_current):
   return {deepcopy(g): deepcopy(p) for g, p in current_GP_dict.items() if min([difference_two_genotypes(g, P_g_current[individual, :]) for individual in range(P_g_current.shape[0])]) <= 4}

def difference_two_genotypes(g1, g2):
   return len([i for i in range(len(g1)) if g1[i] != g2[i]])

#@jit()
def run_WrightFisher(T, N, mu, startgene, phenos_to_track, L, p0, no_generatioons_init = 10):
   print('run_WrightFisher')
   assert get_mfe_structure_as_int(tuple(startgene)) == p0
   K = 4
   current_GP_dict = {tuple(g): get_mfe_structure_as_int(tuple(g)) for g in neighbours_g_nested_list(startgene, K, L)}
   current_GP_dict[tuple(startgene)] = get_mfe_structure_as_int(tuple(startgene))
   assert 2**32 > N
   assert p0 not in phenos_to_track
   # arrays
   times_p_found, times_p_found_corresp_p, genotype_list =  [], [], np.zeros((1000,L), dtype=np.int16)
   P_g_new = np.zeros(shape=(int(N), int(L)), dtype=np.int16) #population genotypes
   P_g_current = np.zeros((N,L), dtype=np.int16) #population genotypes
   P_p_current=np.zeros(N, dtype=np.uint64) #array for phenotypes
   assert p0 < 2**64
   P_fitness_current=np.zeros(N, dtype=np.float32) #array for fitnesses
   # initial conditions
   for individual in range(N):
      for pos in range(L):
         P_g_current[individual, pos] = startgene[pos] #initial condition for population
      P_p_current[individual] = current_GP_dict[startgene]
      P_fitness_current[individual]=fitness_singlepeak(P_p_current[individual], p0)
   # evolution 
   for step in np.arange(0, T + no_generatioons_init * N):
      step_without_init = step - no_generatioons_init * N
      #choose next generation
      if step %(T//100) == 0:
         current_GP_dict = update_GP_dict(current_GP_dict, P_g_current) # remove genotypes above cutoff distance from existing population
      P_g_new = get_new_generation(P_fitness_current, P_g_current, P_g_new, N, mu, K, L)
      for individual in range(N):
         try:
            P_p_current[individual] = current_GP_dict[tuple(P_g_new[individual, :])]
         except KeyError:
            new_str = get_mfe_structure_as_int(tuple(P_g_new[individual, :]))
            current_GP_dict[tuple(P_g_new[individual, :])] = new_str
            P_p_current[individual] = current_GP_dict[tuple(P_g_new[individual, :])]
            assert new_str < 2**64
         P_fitness_current[individual] = fitness_singlepeak(P_p_current[individual], p0)
      if step_without_init >= 0:
         for individual in range(N):
            if P_p_current[individual] != p0:
               if P_p_current[individual] in phenos_to_track and step_without_init >= 0:
                  times_p_found.append(step_without_init)
                  times_p_found_corresp_p.append(int(P_p_current[individual]))
      P_g_current[:,:] = P_g_new[:,:].copy()
      if np.count_nonzero(P_fitness_current)==0:
         raise RuntimeError('The population has left the NC.', flush=True)
      if step_without_init %(T//1000) == 0 and step_without_init >= 0:
         print('finished', step_without_init/float(T) * 100, '%', flush=True)
         genotype_tuples = [tuple(P_g_current[individual,:]) for individual in range(N)]
         median_genotype = max(genotype_tuples, key=Counter(genotype_tuples).get)
         for geno_pos in range(L):
            genotype_list[step_without_init//(T//1000), geno_pos] =  median_genotype[geno_pos]
      elif step_without_init < 0 and step%(T//1000) == 0:
         print('finished', step/float(no_generatioons_init * N) * 100, '% of', no_generatioons_init * N, 'initialising generations', flush=True)
   return times_p_found, times_p_found_corresp_p, genotype_list







