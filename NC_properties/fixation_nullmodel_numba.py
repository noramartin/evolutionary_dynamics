#!/usr/bin/env python
import numpy as np
from numba import jit
import random
from collections import Counter


@jit(nopython=True)
def select_mutation_for_numba_meanfield(mu):
   randomnumber = random.random()
   if randomnumber > mu:
      return False
   else:
      return True

@jit(nopython=True)
def fitness_threepeaks_meanfield(ph, three_peaks, fitness_three_peaks):
   for i in range(3):
      if ph == three_peaks[i]:
         return fitness_three_peaks[i]
   return 0



def get_new_generation_meanfield_nonumba(P_p_current, P_fitness_current, mean_field_rates, P_p_new, N, mu, p0p1p2, fitnessp0p1p2 ):
   p0, p1, p2 = p0p1p2
   p_invalid = 0
   new_p1, new_p2 = 0, 0
   p_to_index = {p0: 0, p1: 1, p2: 2}
   P_list = np.divide(P_fitness_current[:],float(np.sum(P_fitness_current[:])))
   rows = np.random.choice(N, size=N, p=P_list, replace=True) #select N rows that get chosen for the next generation
   for individual in range(N):
      P_p_new[individual] = P_p_current[rows[individual]] #fill in genes for next generation
      #mutate with p=mu
      mutate=select_mutation_for_numba_meanfield(mu)
      if mutate:
         assert P_p_new[individual] != p_invalid #invalid one should go extinct because inviable
         mut_rates = [p for p in mean_field_rates[p_to_index[P_p_new[individual]]]]
         new_p = np.random.choice([p0, p1, p2, p_invalid], p = mut_rates + [1 - sum(mut_rates),])
         if new_p == p1 and P_p_current[rows[individual]] != p1:
            new_p1 += 1
         if new_p == p2 and P_p_current[rows[individual]] != p2:
            new_p2 += 1            
         P_p_new[individual]= new_p
      P_fitness_current[individual] = fitness_threepeaks_meanfield(P_p_new[individual], p0p1p2, fitnessp0p1p2)
   return P_p_new, P_fitness_current, new_p1, new_p2





def run_WrightFisher_fixation_meanfield(T, N, mu, p0p1p2, mean_field_rates, fitnessp0p1p2):
   p0, p1, p2 = p0p1p2
   assert 2**32 > N
   assert 2**63 - 1 > T # limit of np.int64
   # arrays
   #disc_timep1p2 =  np.array([-1, -1], dtype=np.int64),  np.array([-1, -1], dtype=np.int64)
   times_p1_found, times_p2_found =  [], []
   P_p_current=np.zeros(N, dtype=np.uint32) #array for phenotypes
   P_p_new=np.zeros(N, dtype=np.uint32) #array for phenotypes
   P_fitness_current=np.zeros(N, dtype=np.float32) #array for fitnesses
   ### for initial relaxation on network
   fitnessinitialisation = np.array([1, 0, 0])
   # initial conditions
   for individual in range(N):
      P_p_current[individual] = p0
      P_fitness_current[individual] = fitness_threepeaks_meanfield(P_p_current[individual], p0p1p2, fitnessinitialisation) #fitness is single-peaked
   # evolution 
   for step in range(0, int(T)):
      #choose next generation
      P_p_new, P_fitness_current, new_p1, new_p2 = get_new_generation_meanfield_nonumba(P_p_current, P_fitness_current, mean_field_rates, P_p_new, N, mu, p0p1p2, fitnessp0p1p2 )   
      P_p_current[:] = P_p_new[:].copy()
      if new_p1:
         times_p1_found += [step,] * new_p1
      if new_p2:
         times_p2_found+= [step,] * new_p2
      if (P_p_current == p0).sum() < 0.25*N: #fixation
         fixation_time = step    
         phenotypes_present, phenotype_counts = np.unique(P_p_current, return_counts=True) 
         phenotype_fixation = phenotypes_present[np.argmax(phenotype_counts)]
         break 
      if step %(T//1000) == 0:
         print('finished', step/float(T) * 100, '%', flush=True)
   return fixation_time, phenotype_fixation, times_p1_found, times_p2_found



def run_fixation_n_times_meanfield(T, N, mu, p0, p1, p2, mean_field_rates, f0, f1, f2, repetitions):
   print('important: mu here is genome mutation rate mu * L')
   """ run Wright-Fisher until fixation n times"""
   discovery_time_allruns_array=np.zeros((repetitions, 2), dtype='int64')
   fixation_time_allruns_array=np.zeros(repetitions, dtype='uint64')
   fixating_phenotype_allruns_array=np.zeros(repetitions, dtype='uint64')
   run_vs_times_appeared_p1, run_vs_times_appeared_p2 = {}, {}
   assert max(p1, p2)<2**64
   assert T<2**64
   for run_number in range(repetitions):
      print('run number', run_number+1, flush=True)
      fixation_time, phenotype_fixation, times_p1_found, times_p2_found = run_WrightFisher_fixation_meanfield(T, N, mu, np.array([p0, p1, p2], dtype=np.uint32), mean_field_rates, np.array([f0, f1, f2], dtype=np.float32))
      fixation_time_allruns_array[run_number] = fixation_time
      fixating_phenotype_allruns_array[run_number] = phenotype_fixation
      for pheno_index, time_list in enumerate([times_p1_found, times_p2_found]):
          if len(time_list):
             discovery_time_allruns_array[run_number, pheno_index] = min(time_list)
          else:
            discovery_time_allruns_array[run_number, pheno_index] = -1
      run_vs_times_appeared_p1[run_number] = times_p1_found
      run_vs_times_appeared_p2[run_number] = times_p2_found
      #np.save(filename_times_appeared + '_p2_rep'+str(run_number)+'.npy', times_p2_found)
   return discovery_time_allruns_array, fixation_time_allruns_array, fixating_phenotype_allruns_array, run_vs_times_appeared_p1, run_vs_times_appeared_p2
   



