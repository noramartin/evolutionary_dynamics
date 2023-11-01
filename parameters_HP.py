#!/usr/bin/env python3
from copy import deepcopy
#from HP_model.HPfunctions import *
import pandas as pd


n = 5
K = 2
L = n * n
###################################################################################################
GPmap_filename = './HP_data/GPmapHP_L'+str(L)+'.npy'
contact_map_list_filename = './HP_data/contact_map_listHP_L'+str(L)+'.csv'
###################################################################################################
###################################################################################################
###################################################################################################
df_CM = pd.read_csv(contact_map_list_filename)
contact_map_list = [tuple([(int(c0), int(c1)) for c0, c1 in zip(row['lower contacts of structure'].strip('_').split('_'), row['corresponding upper contacts of structure'].strip('_').split('_'))]) if len(row['lower contacts of structure']) > 1 else tuple([]) for rowi, row in df_CM.iterrows()]
contact_map_vs_updown = {deepcopy(contact_map_list[rowi]): [int(ud) for ud in row['up-down string of structure'].strip('_').split('_')] for rowi, row in df_CM.iterrows()}
print('contact maps',  len(contact_map_vs_updown))

