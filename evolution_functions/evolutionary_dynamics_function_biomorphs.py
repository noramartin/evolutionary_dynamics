#!/usr/bin/env python
import numpy as np
from numba import jit, prange
from collections import Counter

@jit(nopython=True)
def select_next_generation_for_numba_biomorphs(N, P_list):
   cumsum_P = np.cumsum(P_list)
   assert abs(cumsum_P[N-1] - 1 ) < 0.01
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
def select_mutation_for_numba_biomorphs(mu):
   randomnumber = np.random.rand()
   if randomnumber > mu:
      return False
   else:
      return True

@jit(nopython=True)
def fitness_singlepeak_biomorphs(ph, p0):
   if ph==p0:
      return 1
   else:
      return 0

@jit(nopython=True)
def get_new_generation_biomorphs(P_p_current, P_fitness_current, P_g_current, GPmap, P_g_new, N, mu, L, p0 , max_index_per_site):
   #choose next generation
   P_list = np.divide(P_fitness_current[:],float(np.sum(P_fitness_current[:])))
   rows=select_next_generation_for_numba_biomorphs(N, P_list=P_list) #select N rows that get chosen for the next generation
   for individual in range(N):
      P_g_new[individual,:] = P_g_current[rows[individual],:] #fill in genes for next generation
      #mutate with p=mu
      for pos in range(L):
         mutate=select_mutation_for_numba_biomorphs(mu)
         if mutate:
            alternative_bases = np.array([b for b in range(P_g_new[individual,pos] - 1, P_g_new[individual,pos] + 2) if b!=P_g_new[individual,pos] and b >= 0 and b < max_index_per_site[pos]])
            P_g_new[individual,pos] = np.random.choice(alternative_bases)
      #fill in phenotype and fitness
      pos_in_geno_space = (P_g_new[individual,0], P_g_new[individual,1], P_g_new[individual,2], P_g_new[individual,3], P_g_new[individual,4], P_g_new[individual,5], P_g_new[individual,6], P_g_new[individual,7], P_g_new[individual,8])
      P_p_current[individual]= GPmap[pos_in_geno_space]
      P_fitness_current[individual] = fitness_singlepeak_biomorphs(P_p_current[individual], p0)
   return P_p_current, P_fitness_current, P_g_new



#@jit()
def run_WrightFisher_biomorphs(T, N, mu, startgene, phenos_to_track, GPmap, p0, no_generatioons_init=10):
   print('run_WrightFisher')
   max_index_per_site = GPmap.shape
   assert 2**32 > N
   assert p0 not in phenos_to_track
   L = 9
   # arrays
   times_p_found, times_p_found_corresp_p, genotype_list =  [], [], np.zeros((1000,L), dtype=np.int16)
   P_g_new = np.zeros(shape=(int(N), int(L)), dtype=np.int16) #population genotypes
   P_g_current = np.zeros((N,L), dtype=np.int16) #population genotypes
   P_p_current=np.zeros(N, dtype=np.uint32) #array for phenotypes
   P_fitness_current=np.zeros(N, dtype=np.float32) #array for fitnesses
   # initial conditions
   for individual in range(N):
      for pos in range(L):
         P_g_current[individual, pos] = startgene[pos] #initial condition for population
      P_p_current[individual] = GPmap[startgene]
      assert P_p_current[individual]  == p0
      P_fitness_current[individual]=fitness_singlepeak_biomorphs(P_p_current[individual], p0)
   # evolution 
   for step in np.arange(0, T + no_generatioons_init * N):
      #choose next generation
      step_without_init = step - no_generatioons_init * N
      P_p_current, P_fitness_current, P_g_new = get_new_generation_biomorphs(P_p_current, P_fitness_current, P_g_current, GPmap, P_g_new, N, mu, L, p0, max_index_per_site )
      if step_without_init >= 0:
         for individual in range(N):
            if P_p_current[individual] != p0:
               for pheno_to_track in phenos_to_track:
                  if P_p_current[individual] == pheno_to_track:
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







