#!/usr/bin/env python3
import numpy as np
from bar_plot_functions import  fraction_at_Hamming_distance_x_randommap_theory
from scipy.special import comb


def const_rate_fixation_time(s, N, cpq, L, mu):
    pfix_single = (1.0-np.exp(-2.0*s))/(1.0-np.exp(-2.0*N*s))
    t_appearance_mean_field = (cpq * L * N * mu)**-1
    t_expected = t_appearance_mean_field/min(pfix_single, 1)
    return t_expected


def bursty_fixation_time(s, N, cpq, L, mu, rho_p0, K, polymorphic_correction=True):
    assert cpq * L * (K-1) < 1 #otherwise p1 is accessible from every genotype
    pfix_single = (1.0-np.exp(-2.0*s))/(1.0-np.exp(-2.0*N*s))
    t_neutral = 1/(mu*L*rho_p0)
    if polymorphic_correction:
        f = fraction_at_Hamming_distance_x_randommap_theory(N, mu, L, rho_p0)
        pfix_burst = (1 + (K-1) * rho_p0 * L/(N * f * pfix_single) )**(-1)
        assert pfix_burst <= 1
        t_appearance_burst = t_neutral/(cpq * L * (K-1))
        t_expected_burst_only = t_appearance_burst/min(pfix_burst, 1)
        t_appearance_polymorphic =  (cpq * L * N * (1-f) * mu)**-1
        t_expected_polymorphic_only = t_appearance_polymorphic/pfix_single
        t_expected = t_expected_polymorphic_only * t_expected_burst_only /(t_expected_polymorphic_only + t_expected_burst_only)
    else:
        pfix_burst = (1 + (K-1) * rho_p0 * L/(N * pfix_single) )**(-1)
        t_appearance_burst = t_neutral/(cpq * L * (K-1))
        t_expected_burst_only = t_appearance_burst/min(pfix_burst, 1)
        if min(pfix_burst, 1) == 1:
            print('bursty fixation probability > 1 -- approximation may not hold')
        t_expected = t_expected_burst_only  	
    return t_expected

