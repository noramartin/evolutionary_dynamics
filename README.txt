This folder contains the code required for the preprint "Non-Poissonian bursts in the arrival of phenotypic variation can strongly affect the dynamics of adaptation" (Nora S Martin, Steffen Schaper, Chico Q. Camargo, Ard A. Louis)

This code uses ViennaRNA (2.4.14) and Python 3.7, various standard Python packages (matplotlib, pandas, networkx, numpy etc.) and numba (0.39.0) to speed up the Wright-Fisher model simulations.


The different scripts focus on the following aspects:
- generate_RNA_map.py to generate the different GP map models.
- test_cpq_identical.py runs some tests on the GP map models.
- run_evolutionary_dynamic_for_bar_plot.py evolutionary dynamics with stabilising selection
- run_evolutionary_dynamics_fixation_plot.py fixation for one or two-peaked landscape (for a one-peaked landscape two phenotypes p1 & p2 still have to be defined, but with one fitness set to zero) 
- simple_fixation_plot.py plots the data for the single-peaked fixation scenario
- plots_arrival_of_frequent_1D.py plots the data for the two-peaked fixation scenario
- plot_three_NCs_as_network.py all network plots
- coeff_variation_plot_new.py for the coefficient of variation plots
- bar_plot.py for the histograms
- fixation_time_functions.py and bar_plot_functions.py implement some of the calculations in the appendix


For other GP maps:
- get_NCdata_HPmodel.py is needed to extract phi_pq and NCs from the HP GP map
- get_NCdata_biomorphs.py is needed to extract phi_pq and NCs from the biomorphs GP map
- bar_plot_othermodels.py to plot the bar plots for the HP model and biomorphs

For longer lengths (without exhaustive GP map knolwdge):
- phi_pq_data_long_seq.py get a sample of structures with a range of numbers of stacks, and estimate phi_pq
- evolutionary_dynamics_function_long_seq.py contains the functions needed for Wright-Fisher dynamics when the GP map is not provided in advance (which would be infeasible for L=30)
- run_evolutionary_dynamic_long_seq.py uses the functions in evolutionary_dynamics_function_long_seq.py and saves the results

Supplementary analyses:
- fixation_single_mutations_or_group.py - does it matter if the introductions of p1 fall into a single generation
- population_diversity_test.py - tests our approximation of mildly polymorphic populations



References:
- ViennaRNA for RNA structure prediction (Lorenz, Ronny, et al. "ViennaRNA Package 2.0." Algorithms for molecular biology 6 (2011): 1-14.)
- Own implementation of the community detection algorithm in Weiß, Marcel, and Sebastian E. Ahnert. "Neutral components show a hierarchical community structure in the genotype–phenotype map of RNA secondary structure." Journal of The Royal Society Interface 17.171 (2020): 20200608.
- Own implementation of randomised GP maps from Schaper, Steffen, and Ard A. Louis. "The arrival of the frequent: how bias in genotype-phenotype maps can steer populations to local optima." PloS one 9.2 (2014): e86635 and from McCandlish, David M. Evolution on Arbitrary Fitness Landscapes When Mutation is Weak. Diss. 2012.
- The functions and data for the HP protein GP map are from Martin, Nora S., and Sebastian E. Ahnert. "The Boltzmann distributions of folded molecular structures predict likely changes through random mutations." bioRxiv (2023): 2023-02.
- The functions and data for the biomorphs GP map are from Martin, Nora S., Chico Q. Camargo, and Ard A. Louis. "Bias in the arrival of variation can dominate over natural selection in Richard Dawkins' biomorphs." bioRxiv (2023): 2023-05.
- RNA sequences of length L=30: The methods used to generate a sample of structures q and estimate the corresponding \phi_pq values closely follow the methods in previous work (Martin, Nora S., and Sebastian E. Ahnert. "The Boltzmann distributions of folded molecular structures predict likely changes through random mutations." bioRxiv (2023): 2023-02) and the site-scanning sampling method (Weiß, Marcel, and Sebastian E. Ahnert. "Using small samples to estimate neutral component size and robustness in the genotype–phenotype map of RNA secondary structure." Journal of the Royal Society Interface 17.166 (2020): 20190784).