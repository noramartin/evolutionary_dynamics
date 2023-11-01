import numpy as np 
from itertools import product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from copy import deepcopy


direction_to_int = {(1, 0): 0, (-1, 0): 1, (0, 1): 2, (0, -1): 3}
int_to_direction = {i: d for d, i in direction_to_int.items()}
index_to_new_index_90deg = {old_index: direction_to_int[(-old_dir[1], old_dir[0])] for old_dir, old_index in direction_to_int.items()} #rotation by 90
mirror_image_x_axis = {0: 0, 1: 1, 2: 3, 3: 2}
contact_energies = {(0, 0): -1, (0, 1): 0, (1, 0): 0, (1, 1): 0} # H is 0

def enumerate_all_structures(n):
	L = n**2
	structures = []
	for startposition in product(np.arange(n), repeat=2):
		if sum(startposition)%2 == 0 or n%2 == 0: 
		    # only even parity start positions if odd n (source: Sam Greenbury's thesis who relies on Kloczkowski and Jernigan 1997)
		    # a simple way of seeing that this has to hold: an odd-length chain will start and end on the same parity, 
		    # but a 5x5 lattice has more sites of even parity -> need to start and end at even parity
		   	structures += find_all_walks_given_start(tuple(startposition), n, L, [startposition,])
	structures_up_down_notation = take_out_mirror_images(list(set([from_coordinates_to_up_down(s, L) for s in structures])))
	return structures_up_down_notation

def take_out_mirror_images(structure_list):
	#a flip across x-axis is same structure, just turned
	mirror_images = [tuple([mirror_image_x_axis[i] for i in s]) for s in structure_list] # the mirrored structures still have their first direction aligned with the x-axis because we flip over x
	structure_list_no_mirror_images = []
	for i, s in enumerate(structure_list):
		if s not in mirror_images[:i]:
		    structure_list_no_mirror_images.append(s)
	return structure_list_no_mirror_images


def turn_such_that_first_index_zero(structure_up_down_notation):
	while structure_up_down_notation[0] != 0:
		structure_up_down_notation = [index_to_new_index_90deg[i] for i in structure_up_down_notation]
	return tuple(structure_up_down_notation)

def from_coordinates_to_up_down(s, L):
	return turn_such_that_first_index_zero([direction_to_int[(s[i + 1][0] - s[i][0], s[i + 1][1] - s[i][1])]  for i in range(L - 1)])


def find_all_walks_given_start(startposition, n, L, walk_so_far):
	if len(walk_so_far) == L:
		return [deepcopy(walk_so_far), ]
	list_walks = []
	for next_move in ((1, 0), (-1, 0), (0, 1), (0, -1)):
		new_pos = (startposition[0] + next_move[0], startposition[1] + next_move[1]) 
		if max(new_pos) <= n - 1 and min(new_pos) >= 0 and new_pos not in walk_so_far:
			list_walks += find_all_walks_given_start(new_pos, n, L, deepcopy(walk_so_far + [new_pos,]))
	return list_walks

def all_structures_with_unique_contact_maps(n):
	structures_up_down_notation = enumerate_all_structures(n)
	contact_maps = [contact_map_to_str(up_down_to_contact_map(s)) for s in structures_up_down_notation]
	return [deepcopy(structures_up_down_notation[i]) for i, c in enumerate(contact_maps) if contact_maps.count(c) == 1]


def plot_structure(structure_up_down_notation, ax, redpointsize=6):
	ax.axis('off')
	ax.axis('equal')
	current_point = (0, 0)
	ax.scatter([current_point[0], ], [current_point[1],], c='r', s=redpointsize)
	for d in structure_up_down_notation:
		new_pos = (current_point[0] + int_to_direction[d][0], current_point[1] + int_to_direction[d][1]) 
		ax.plot([current_point[0], new_pos[0]], [current_point[1], new_pos[1]], c='k')
		current_point = (new_pos[0], new_pos[1])

def up_down_to_contact_map(structure_up_down_notation):
	current_point = (0, 0)
	structure_coordinate_notation = [(0, 0)]
	for d in structure_up_down_notation:
		current_point = (current_point[0] + int_to_direction[d][0], current_point[1] + int_to_direction[d][1]) 
		structure_coordinate_notation.append((current_point[0], current_point[1]))
	contact_map = []
	for i, coordi in enumerate(structure_coordinate_notation):
		for j, coordj in enumerate(structure_coordinate_notation):
			if i < j - 1.5 and abs(coordi[0] - coordj[0]) + abs(coordi[1] - coordj[1]) == 1:
			   contact_map.append((i, j))
	return tuple(sorted(contact_map))

def contact_map_to_str(cm):
	return '__'.join([str(i) + '_' + str(j) for i, j in cm])

def free_energy(seq, contact_map):
	return sum([contact_energies[(seq[i], seq[j])] for i, j in contact_map])


def find_mfe(seq, contact_map_list):
	# if unique ground state exists, return index of its contact map +1 - else return 0
	free_energy_list = [free_energy(seq, contact_map) for contact_map in contact_map_list]
	sorted_free_energy_list = sorted(free_energy_list)
	if sorted_free_energy_list[0] < sorted_free_energy_list[1] - 0.00001:
	   return np.argmin(free_energy_list) + 1
	else:
		return 0

def HPget_Boltzmann_freq_list(seq, contact_map_list, kbT, renormalise = False):
	free_energy_list = [free_energy(seq, contact_map) for contact_map in contact_map_list]
	if renormalise:
		min_G = min(free_energy_list)
		exp_list = np.exp(-1.0 * (np.array(free_energy_list) - min_G)/kbT)
	else:
	    exp_list = np.exp(-1.0 * np.array(free_energy_list)/kbT)
	Z = np.sum(exp_list)
	return [e/Z for e in exp_list]

def CMO(cm1, cm2):
	#compute contact map overlap
	return len([c for c in cm1 if c in cm2])

def pattern_internal_surface(contact_map, L):
    site_vs_contacts = {i: 0 for i in range(L)}
    for i, j in contact_map:
        site_vs_contacts[i] += 1
        site_vs_contacts[j] += 1
    return [1 if ((site_vs_contacts[i]> 1 and 0 < i < L-1 ) or site_vs_contacts[i]> 2) else 0 for i in range(L)] # 0 is surface


############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
	n = 3
	all_structures = enumerate_all_structures(n)
	f, ax = plt.subplots(ncols = 4, nrows = len(all_structures)//4 + 1, figsize = (8, len(all_structures)//2) )
	for i, s in enumerate(all_structures):
		plot_structure(s, ax[i//4, i%4])
	f.tight_layout()
	f.savefig('./test_HPstructures'+str(n)+'.png')

	# same for unique CMs
	all_structures = all_structures_with_unique_contact_maps(n)
	f, ax = plt.subplots(ncols = 4, nrows = len(all_structures)//4 + 1, figsize = (8, len(all_structures)//2) )
	for i, s in enumerate(all_structures):
		plot_structure(s, ax[i//4, i%4])
		ax[i//4, i%4].set_title(contact_map_to_str(up_down_to_contact_map(s)), fontsize=6)
	f.tight_layout()
	f.savefig('./test_HPstructures_unique'+str(n)+'.png')
	# try one sequence - example from Sam Greenbury's thesis
	contact_map_list = [up_down_to_contact_map(s) for s in all_structures]
	for seq in ((1, 0, 1, 0, 1, 0, 1, 1, 0), (1, 1, 1, 0, 0, 1, 1, 0, 1)): # P in 1
		print('seq', seq)
		mfe_structure = find_mfe(seq, contact_map_list= contact_map_list)
		if mfe_structure == 0:
			print( 'no unique mfe structure')
			print('free energy of all structures', [free_energy(seq, c) for c in contact_map_list])
		f, ax = plt.subplots(figsize = (3, 3) )
		plot_structure(all_structures[mfe_structure -1], ax)
		ax.set_title('seq = ' + ''.join([{0: 'H', 1: 'P'}[i] for i in seq]))
		f.tight_layout()
		f.savefig('./mfe_structure'+str(n)+'_'+str(''.join([str(c) for c in seq]))+'.png')
		# check contact map
		print('contact map of mfe structure', up_down_to_contact_map(all_structures[mfe_structure -1]))
		print('free energy of mfe structure', free_energy(seq, up_down_to_contact_map(all_structures[mfe_structure -1])))
		print('free energy of all structures', [free_energy(seq, c) for c in contact_map_list])
		print('Boltzmann factors', HPget_Boltzmann_freq_list(seq, contact_map_list=contact_map_list, kbT=1), '\n\n')
	###



	

