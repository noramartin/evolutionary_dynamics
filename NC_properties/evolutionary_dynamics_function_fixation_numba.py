#!/usr/bin/env python
import numpy as np
from numba import jit
import random
from collections import Counter

@jit(nopython=True)
def select_next_generation_for_numba(N, P_list):
   cumsum_P = np.cumsum(P_list)
   selected_index = np.zeros(N, dtype=np.uint32)
   for i in range(N):
      randomnumber = random.random()
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
def select_mutation_for_numba( mu):
   randomnumber = random.random()
   if randomnumber > mu:
      return False
   else:
      return True

@jit(nopython=True)
def fitness_threepeaks(ph, three_peaks, fitness_three_peaks):
   for i in range(3):
      if ph == three_peaks[i]:
         return fitness_three_peaks[i]
   return 0

@jit(nopython=True)
def get_new_generation(P_p_current, P_fitness_current, P_g_current, GPmap, P_g_new, P_p_new, N, mu, K, L, p0p1p2, fitnessp0p1p2 ):
   p0, p1, p2 = p0p1p2
   new_p1, new_p2 = 0, 0
   P_list = np.divide(P_fitness_current[:],float(np.sum(P_fitness_current[:])))
   rows=select_next_generation_for_numba(N, P_list=P_list) #select N rows that get chosen for the next generation
   for individual in range(N):
      P_g_new[individual,:] = P_g_current[rows[individual],:] #fill in genes for next generation
      #mutate with p=mu
      for pos in range(L):
         mutate=select_mutation_for_numba(mu)
         if mutate:
            alternative_bases = np.array([b for b in range(K) if b!=P_g_new[individual,pos]])
            randomindex = random.randrange(0, len(alternative_bases))
            P_g_new[individual,pos] = alternative_bases[randomindex]
      #fill in phenotype and fitness
      pos_in_geno_space = (P_g_new[individual,0], P_g_new[individual,1], P_g_new[individual,2], P_g_new[individual,3], P_g_new[individual,4], P_g_new[individual,5], P_g_new[individual,6], P_g_new[individual,7], P_g_new[individual,8], P_g_new[individual,9], P_g_new[individual,10], P_g_new[individual,11])
      if GPmap[pos_in_geno_space] == p1 and P_p_current[rows[individual]] != p1:
         new_p1 += 1
      if GPmap[pos_in_geno_space] == p2 and P_p_current[rows[individual]] != p2:
         new_p2 += 1
      P_p_new[individual]= GPmap[pos_in_geno_space]
      P_fitness_current[individual] = fitness_threepeaks(P_p_new[individual], p0p1p2, fitnessp0p1p2)
   return P_p_new, P_fitness_current, P_g_new, new_p1, new_p2


def run_WrightFisher_fixation(T, N, mu, startgene, p0p1p2, GPmap, fitnessp0p1p2, no_generatioons_no_fix):
   p0, p1, p2 = p0p1p2
   K, L = int(GPmap.shape[1]), int(GPmap.ndim)
   assert 2**32 > N
   assert 2**63 - 1 > T # limit of np.int64
   # arrays
   #disc_timep1p2 =  np.array([-1, -1], dtype=np.int64),  np.array([-1, -1], dtype=np.int64)
   times_p1_found, times_p2_found, genotype_list =  [], [], np.zeros((1000,L), dtype=np.int16)
   P_g_new = np.zeros(shape=(int(N), int(L)), dtype=np.int16) #population genotypes
   P_g_current = np.zeros((N,L), dtype=np.int16) #population genotypes
   P_p_current=np.zeros(N, dtype=np.uint32) #array for phenotypes
   P_p_new=np.zeros(N, dtype=np.uint32) #array for phenotypes
   P_fitness_current=np.zeros(N, dtype=np.float32) #array for fitnesses
   ### for initial relaxation on network
   fitnessinitialisation = np.array([1, 0, 0])
   # initial conditions
   for individual in range(N):
      for pos in range(L):
         P_g_current[individual, pos] = startgene[pos] #initial condition for population
      P_p_current[individual] = GPmap[startgene]
      P_fitness_current[individual] = fitness_threepeaks(P_p_current[individual], p0p1p2, fitnessinitialisation) #fitness is single-peaked
      assert P_p_current[individual] == p0 and abs(P_fitness_current[individual] - 1) < 0.01
   # initialisation 
   for step in range(0, int(no_generatioons_no_fix * N)):
      #choose next generation
      P_p_new, P_fitness_current, P_g_new, new_p1, new_p2 = get_new_generation(P_p_current, P_fitness_current, P_g_current, GPmap, P_g_new, P_p_new, N, mu, K, L, p0p1p2, fitnessinitialisation )   
      P_g_current[:,:] = P_g_new[:,:].copy()
      P_p_current[:] = P_p_new[:].copy()
   # evolution 
   for step in range(0, int(T)):
      #choose next generation
      P_p_new, P_fitness_current, P_g_new, new_p1, new_p2 = get_new_generation(P_p_current, P_fitness_current, P_g_current, GPmap, P_g_new, P_p_new, N, mu, K, L, p0p1p2, fitnessp0p1p2 )   
      P_g_current[:,:] = P_g_new[:,:].copy()
      P_p_current[:] = P_p_new[:].copy()
      if new_p1:
         times_p1_found += [step,] * new_p1
      if new_p2:
         times_p2_found += [step,] * new_p2
      if (P_p_current == p0).sum() < 0.25*N: #fixation
         fixation_time = step    
         phenotypes_present, phenotype_counts = np.unique(P_p_current, return_counts=True) 
         phenotype_fixation = phenotypes_present[np.argmax(phenotype_counts)]
         return fixation_time, phenotype_fixation, times_p1_found, times_p2_found, genotype_list 
      if step %(T//1000) == 0:
         print('finished', step/float(T) * 100, '%', flush=True)
         genotype_tuples = [tuple(P_g_current[individual,:]) for individual in range(N)]
         median_genotype = max(genotype_tuples, key=Counter(genotype_tuples).get)
         for geno_pos in range(L):
            genotype_list[step//(T//1000), geno_pos] =  median_genotype[geno_pos]
   return -1, -1, times_p1_found, times_p2_found, genotype_list




def run_fixation_n_times(T, N, mu, startgene_list, p0, p1, p2, GPmap, f0, f1, f2, repetitions, no_generatioons_no_fix = 10):
   """ run Wright-Fisher until fixation n times"""
   discovery_time_allruns_array=np.zeros((repetitions, 2), dtype='int64')
   fixation_time_allruns_array=np.zeros(repetitions, dtype='uint64')
   fixating_phenotype_allruns_array=np.zeros(repetitions, dtype='uint64')
   run_vs_times_appeared_p1, run_vs_times_appeared_p2, run_vs_genotype_list = {}, {}, {}
   assert max(p1, p2)<2**64
   assert T<2**64
   for run_number in range(repetitions):
      print('run number', run_number+1, flush=True)
      assert p0 == GPmap[tuple(startgene_list[run_number])]
      fixation_time, phenotype_fixation, times_p1_found, times_p2_found, genotype_list = run_WrightFisher_fixation(T, N, mu, tuple(startgene_list[run_number]), np.array([p0, p1, p2], dtype=np.uint32), GPmap, np.array([f0, f1, f2], dtype=np.float32), no_generatioons_no_fix)
      fixation_time_allruns_array[run_number] = fixation_time
      fixating_phenotype_allruns_array[run_number] = phenotype_fixation
      for pheno_index, time_list in enumerate([times_p1_found, times_p2_found]):
          if len(time_list):
             discovery_time_allruns_array[run_number, pheno_index] = min(time_list)
          else:
            discovery_time_allruns_array[run_number, pheno_index] = -1
      run_vs_times_appeared_p1[run_number] = times_p1_found
      run_vs_times_appeared_p2[run_number] = times_p2_found
      run_vs_genotype_list[run_number] = genotype_list
      #np.save(filename_times_appeared + '_p2_rep'+str(run_number)+'.npy', times_p2_found)
   return discovery_time_allruns_array, fixation_time_allruns_array, fixating_phenotype_allruns_array, run_vs_times_appeared_p1, run_vs_times_appeared_p2, run_vs_genotype_list
   



def find_startgeno_randommap(GPmap, p0):
   startgene_found = False
   while not startgene_found:
      trial_gene_list = [random.randrange(0, GPmap.shape[0]) for pos in range(GPmap.ndim)]
      if GPmap[tuple(trial_gene_list)] == p0:
         startgene = tuple(np.copy(trial_gene_list))
         startgene_found=True
   assert GPmap[startgene]==p0
   return tuple(startgene)

############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   import pandas as pd

   s1, s2, type_map = 0.005, 0.005, 'randommap'
   N, mu, T, repetitions = 500, 0.0002, 10**8, 200
   p0, p1, p2, f0, f1, f2, K, L = 27263296, 27788608, 19398976, 1.0, 1.0 + s1, 1.0 + s2, 4, 12
   NCindex = 171
   NCindex_arrayRNA = np.load('./RNA_data/RNA_NCindexArray_L'+str(L)+'.npy')
   print( 's1: '+str(s1), ', s2: '+str(s2), type_map, 'N', N, 'mu', mu, 'T', T, 'no_runs', repetitions)
   GPmapRNA=np.load('./RNA_data/RNA_GPmap_L'+str(L)+'.npy')
   V = np.argwhere(NCindex_arrayRNA==NCindex)
   startgene_list = [tuple(V[random.randrange(0, V.shape[0]),:]) for i in range(repetitions)]
   print(startgene_list[0])
   discovery_time_allruns_array, fixation_time_allruns_array, fixating_phenotype_allruns_array, run_vs_times_appeared_p1, run_vs_times_appeared_p2 = run_fixation_n_times(T, N, mu, startgene_list, p0, p1, p2, GPmapRNA, f0, f1, f2, repetitions, no_generatioons_no_fix = 2)
   print('discovery_time_allruns_array', discovery_time_allruns_array)
   print('fixation', fixation_time_allruns_array, fixating_phenotype_allruns_array)
   #print('\np1 appears', run_vs_times_appeared_p1)
   #print('\np2 appears', run_vs_times_appeared_p2)
   number_p1s = sum([len(v) for v in run_vs_times_appeared_p1.values()])
   number_p2s = sum([len(v) for v in run_vs_times_appeared_p2.values()])
   #### check against cpq
   filename_cpq = './RNA_data/RNA_NCs_cqp_data_L'+str(L)+'.csv'
   df_cpq, NC_vs_p_vs_cpq = pd.read_csv(filename_cpq), {}
   for i, row in df_cpq.iterrows():
       try:
          NC_vs_p_vs_cpq[row['NCindex']][int(row['new structure integer'])] = row['cpq']
       except KeyError:
          NC_vs_p_vs_cpq[row['NCindex']] = {int(row['new structure integer']): row['cpq']}
   print('expected ratio:', NC_vs_p_vs_cpq[NCindex][p1]/NC_vs_p_vs_cpq[NCindex][p2])
   print('simulation ratio:', number_p1s/number_p2s, '- based on numbers: ', number_p1s, number_p2s)


