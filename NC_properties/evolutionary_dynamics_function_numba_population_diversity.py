#!/usr/bin/env python
import numpy as np
from numba import jit, prange
from collections import Counter

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
            selected_ind = ind
         elif ind == 0 and 0 <= randomnumber < cumsum_P[ind]:
            assert selected_ind == 2*N # check that only one individual meets condition
            selected_ind = ind
         elif ind == N-1 and cumsum_P[ind-1] <= randomnumber:
            assert selected_ind == 2*N # check that only one individual meets condition
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
def get_new_generation(P_p_current, P_fitness_current, P_g_current, GPmap, P_g_new, N, mu, K, L, p0 ):
   #choose next generation
   P_list = np.divide(P_fitness_current[:],float(np.sum(P_fitness_current[:])))
   rows=select_next_generation_for_numba(N, P_list=P_list) #select N rows that get chosen for the next generation
   for individual in range(N):
      P_g_new[individual,:] = P_g_current[rows[individual],:] #fill in genes for next generation
      #mutate with p=mu
      for pos in range(L):
         mutate=select_mutation_for_numba(mu)
         if mutate:
            alternative_bases = np.array([b for b in range(K) if b!=P_g_new[individual,pos]])
            P_g_new[individual,pos] = np.random.choice(alternative_bases)
      #fill in phenotype and fitness
      pos_in_geno_space = (P_g_new[individual,0], P_g_new[individual,1], P_g_new[individual,2], P_g_new[individual,3], P_g_new[individual,4], P_g_new[individual,5], P_g_new[individual,6], P_g_new[individual,7], P_g_new[individual,8], P_g_new[individual,9], P_g_new[individual,10], P_g_new[individual,11])
      P_p_current[individual]= GPmap[pos_in_geno_space]
      P_fitness_current[individual] = fitness_singlepeak(P_p_current[individual], p0)
   return P_p_current, P_fitness_current, P_g_new


#@jit()
def run_WrightFisher(T, N, mu, startgene, GPmap, p0, no_generatioons_init = 10):
   print('run_WrightFisher')
   K, L = int(GPmap.shape[1]), int(GPmap.ndim)
   assert 2**32 > N
   # arrays
   fraction_at_most_freq_list =  []
   P_g_new = np.zeros(shape=(int(N), int(L)), dtype=np.int16) #population genotypes
   P_g_current = np.zeros((N,L), dtype=np.int16) #population genotypes
   P_p_current=np.zeros(N, dtype=np.uint32) #array for phenotypes
   P_fitness_current=np.zeros(N, dtype=np.float32) #array for fitnesses
   # initial conditions
   for individual in range(N):
      for pos in range(L):
         P_g_current[individual, pos] = startgene[pos] #initial condition for population
      P_p_current[individual] = GPmap[startgene]
      P_fitness_current[individual]=fitness_singlepeak(P_p_current[individual], p0)
   # evolution 
   for step in np.arange(0, T + no_generatioons_init * N):
      #choose next generation
      P_p_current, P_fitness_current, P_g_new = get_new_generation(P_p_current, P_fitness_current, P_g_current, GPmap, P_g_new, N, mu, K, L, p0 )
      P_g_current[:,:] = P_g_new[:,:].copy()
      if np.count_nonzero(P_fitness_current)==0:
         raise RuntimeError('The population has left the NC.', flush=True)
      if step %(T//1000) == 0 and step > no_generatioons_init * N:
         print('finished', step/float(T) * 100, '%', flush=True)
         genotype_tuples = [tuple(P_g_current[individual,:]) for individual in range(N)]
         fraction_at_most_freq_list.append(max([c for c in Counter(genotype_tuples).values()])/float(N))
   return fraction_at_most_freq_list







