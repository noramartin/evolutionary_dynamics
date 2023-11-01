#!/usr/bin/env python3
import numpy as np
from scipy.stats import poisson, binom



def fraction_at_Hamming_distance_x_randommap_theory(N, mu, L, rho_p0):
   return np.sqrt(1/(1+2 * N * (1- mu * L * (1-rho_p0)) * mu * L * rho_p0))
   


def prob_distribution(M, f, tau, mu, N, K, L, rho_p0, t_neutral, polymorphic_correction=False):
   """ calculate predicted distribution (dynamic bursts, but no static bursts)"""
   if polymorphic_correction:
      fraction_at_zero = fraction_at_Hamming_distance_x_randommap_theory(N, mu, L, rho_p0) #take square root because otherwise gives probability that two individuals have H=0 to each other; 
   else:
      fraction_at_zero = 1.0
   ###
   sum_p=0.0
   for n_p in range(0, (K-1)*L+1):
      f_p_Hdist1 = n_p/float((K-1)*L)
      average_production_in_tau = (f_p_Hdist1 * fraction_at_zero + (1 - fraction_at_zero ) * f) * mu* N * L * tau 
      sum_p += binom.pmf(n_p, (K-1)*L, f, loc=0) * poisson.pmf(M, average_production_in_tau) 
   return sum_p * np.exp(-0.5 * tau/t_neutral) + poisson.pmf(M, f*tau*L*mu * N)* (1 - np.exp(-0.5 * tau/t_neutral)) 


def bar_chart_counts_from_discovery_time_list(disc_list, tau, Tmax):
   """get event counts over intermediate time scale from list of discovery times"""
   mutants_over_steps_of_tau_dict = {index: 0 for index in range(1+int(Tmax//tau))}
   for disc_time in disc_list:
      mutants_over_steps_of_tau_dict[int(disc_time//tau)] += 1
   return np.array([mutants_over_steps_of_tau_dict[index] for index in range(int(Tmax//tau))], dtype='uint32')


