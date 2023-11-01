import pandas as pd


L = 12
##################################################################################################
##################################################################################################
print("larger set")
##################################################################################################
##################################################################################################
df = pd.read_csv('./RNA_data/RNA_NCs_L'+str(L)+'_forrandommap.csv')
NCindex_list = df['NCindex'].tolist()
NCindex_vs_structure = {NCindex: structure_int for NCindex, structure_int in zip(df['NCindex'].tolist(), df['structure integer'].tolist())}
NCindex_vs_seq = {NCindex: seq for NCindex, seq in zip(df['NCindex'].tolist(), df['example sequence'].tolist())}
##################################################################################################
##################################################################################################
print("for everything except two-peaked landscape")
##################################################################################################
##################################################################################################
NCindex_first_plots = 196
seq_firstplots = NCindex_vs_seq[NCindex_first_plots]
pg_first_plots = NCindex_vs_structure[NCindex_first_plots]
pb_first_plots = 19529749
pr_first_plots = 19530064

##################################################################################################
##################################################################################################
print("for two-peaked landscape")
##################################################################################################
##################################################################################################
seq_twopeaked = 'ACCUAAAAAAGG'
p0_twopeaked = 19529749
p1_twopeaked = 27262981
p2_twopeaked = 27918661
