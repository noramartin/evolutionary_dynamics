try:
   import RNA
except ImportError:
   import sys
   #sys.path.insert(0,'/home/nora/bin/lib/python2.7/site-packages') #for ViennaRNA
   sys.path.insert(0,'/mnt/zfsusers/nmartin/ViennaRNA/lib/python3.6/site-packages')
   import RNA

index_to_base = {0: 'A', 1: 'C', 2: 'U', 3: 'G'}
base_to_number={'A':0, 'C':1, 'U':2, 'G':3, 'T':2}
db_to_bin = {'.': '00', '(': '10', ')': '01', '_': '00', '[': '10', ']': '01'}

RNA.cvar.uniq_ML = 1


def get_mfe_structure_as_int(sequence_indices):
   return dotbracket_to_int(get_mfe_structure(sequence_indices))


def get_mfe_structure(sequence_indices, check_degeneracy = True):
   a = RNA.fold_compound(sequence_int_to_str(sequence_indices))
   (mfe_structure, mfe) = a.mfe()
   if check_degeneracy:
      structure_vs_energy = {}
      a.subopt_cb(int(0.11*100.0), convert_result_to_dict, structure_vs_energy)
      alternative_s_list = [s for s in structure_vs_energy if mfe_structure != s and abs(a.eval_structure(s) - mfe) < 0.01]
      if len(alternative_s_list) > 0:
         #print(seq, alternative_s_list[0],  mfe_structure)
         assert alternative_s_list[0] != mfe_structure and abs(a.eval_structure(alternative_s_list[0]) - a.eval_structure(mfe_structure)) < 0.01
         return '.' #return unfolded if degenerate
   return mfe_structure


def convert_result_to_dict(structure, energy, data):
   """ function needed for subopt from ViennaRNA documentation: https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/RNAlib-2.4.14.pdf"""
   if not structure == None:
      data[structure] = energy   


###############################################################################################
## converting between sequence and structure representations
###############################################################################################
def dotbracket_to_binary(dotbracketstring):
   """translating each symbol in the dotbracket string a two-digit binary number;
   translation is defined such that the number of '1' digits is the number of paired positions"""
   binstr = '1'
   for char in dotbracketstring:
      binstr = binstr + db_to_bin[char]
   return binstr

def dotbracket_to_int(dotbracketstring, check_size_limit=False):
   """convert dotbracket format to integer format:
   this is achieved by translating each symbol into a two-digit binary number
   and then converting this into a base-10 integer; a leading '1' is added, so that leading '0's in the binary string matter for the decimal
   if check_size_limit, the function tests whether the integer is within the range of the numpy uint32 datatype"""
   binstr = dotbracket_to_binary(dotbracketstring)
   if check_size_limit:
      assert len(binstr)<32
   integer_rep = int(binstr, 2)
   if check_size_limit:
      assert 0 < integer_rep < 4294967295 #limit of uint32
   return integer_rep

def sequence_str_to_int(sequence):
   """convert between sequence representations:
   from biological four-letter string to a tuple of integers from 0-3;
   motive: these integers can be used as array tuples"""
   return tuple([base_to_number[b] for b in sequence])

def sequence_int_to_str(sequence_indices_tuple):
   """convert between sequence representations:
   from tuple of integers from 0-3 to biological four-letter string;
   motive: these integers can be used as array tuples"""
   return ''.join([index_to_base[ind] for ind in sequence_indices_tuple])


def get_dotbracket_from_int(structure_int):
   """ retrieve the full dotbracket string from the integer representation"""
   dotbracketstring = ''
   bin_to_db = {'10': '(', '00': '.', '01': ')'}
   structure_bin = bin(structure_int)[3:] # cut away '0b' and the starting 1
   assert len(structure_bin) % 2 == 0
   for indexpair in range(0, len(structure_bin), 2):
      dotbracketstring = dotbracketstring + bin_to_db[structure_bin[indexpair]+structure_bin[indexpair+1]]
   return dotbracketstring


def isundefinedpheno(structure_int):
   """ check from the integer structure representation if a structure 
   is the "undefined"/unfolded structure (no base pairs)"""
   if structure_int > 1:
      structure_bin = bin(structure_int)[3:] # cut away '0b' and the starting 1
      return '1' not in structure_bin
   else: # unstable structure in $deltaG<0$ map is '0'
      return True

def dotbracket_to_coarsegrained(dotbracketstring):
   fine_grained_to_coarse_grained_symbol = {'(': '[', ')': ']', '.': '_'}
   basepair_index_mapping = get_basepair_indices_from_dotbracket(dotbracketstring)
   coarse_grained_string = ''
   for charindex, char in enumerate(dotbracketstring):
      if charindex == 0  or dotbracketstring[charindex-1] != dotbracketstring[charindex]:
         coarse_grained_string = coarse_grained_string + fine_grained_to_coarse_grained_symbol[char]
      elif dotbracketstring[charindex-1] == dotbracketstring[charindex] and dotbracketstring[charindex] != '.': #two subsequent brackets of same type
         if not abs(basepair_index_mapping[charindex]-basepair_index_mapping[charindex-1])<1.5:
            coarse_grained_string = coarse_grained_string + fine_grained_to_coarse_grained_symbol[char]
         else:
            pass
      else:
         pass
   return coarse_grained_string

def get_basepair_indices_from_dotbracket(dotbracketstring):
   assert '[' not in dotbracketstring
   base_pair_mapping = {}
   number_open_brackets = 0
   opening_level_vs_index = {}
   for charindex, char in enumerate(dotbracketstring):
      if char == '(':
         number_open_brackets += 1
         opening_level_vs_index[number_open_brackets] = charindex
      elif char == ')':
         base_pair_mapping[charindex] = opening_level_vs_index[number_open_brackets]
         base_pair_mapping[opening_level_vs_index[number_open_brackets]] = charindex
         del opening_level_vs_index[number_open_brackets]
         number_open_brackets -= 1
      elif char == '.':
         pass
      else:
         raise ValueError('invalid character in dot-bracket string')
      if number_open_brackets < 0:
         raise ValueError('invalid dot-bracket string')
   if number_open_brackets != 0:
      raise ValueError('invalid dot-bracket string')
   return base_pair_mapping
#############################################################################################
if __name__ == "__main__":
   import numpy as np
   ############################################################################################################
   print('\n-------------\n\ntest structure conversion')
   ############################################################################################################
   seq_len = 20
   for test_no in range(10**4):
      seq = ''.join(np.random.choice(['A', 'C', 'G', 'U'], seq_len, replace=True))
      mfe_struct = get_mfe_structure(sequence_str_to_int(seq))   
      if mfe_struct == '.':
        assert isundefinedpheno(dotbracket_to_int(mfe_struct)) 
        continue
      assert dotbracket_to_int(mfe_struct, check_size_limit=False) == get_mfe_structure_as_int(sequence_str_to_int(seq))
      assert get_dotbracket_from_int(dotbracket_to_int(mfe_struct, check_size_limit=False)) == mfe_struct
      assert sequence_int_to_str(sequence_str_to_int(seq)) == seq
   ############################################################################################################
   print('\n-------------\n\ntest coarse-graining')
   ############################################################################################################
   assert dotbracket_to_coarsegrained('...(..((....))..)...') == '_[_[_]_]_'
   assert dotbracket_to_coarsegrained('..(((((....).))))...') == '_[[_]_]_'
   assert dotbracket_to_coarsegrained('...(((...))).(.(...))..') == '_[_]_[_[_]]_'
   assert dotbracket_to_coarsegrained('...(((...)))(.(...))..') == '_[_][_[_]]_'
   assert dotbracket_to_coarsegrained('(((...)))(.(...))..') == '[_][_[_]]_'
   assert dotbracket_to_coarsegrained('...(((...)))(.(...))') == '_[_][_[_]]'
