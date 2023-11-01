import numpy as np

def format_integer_filename(n):
   if n < 0:
   	 initial_sign, n = 'minus', np.abs(n)
   else:
   	  initial_sign = ''
   x = int(str(n).rstrip('0'))
   return initial_sign+ str(x)+'_'+str(int(round(np.log10(n/x))))

def format_float_filename(n):
   if n == -1:
      return 'minus1_0' 
   assert -1 < n < 1
   if n < 0:
   	 initial_sign, n = 'minus', np.abs(n)
   else:
   	  initial_sign = ''
   x = int(format(n * 10**(-1 * int(np.log10(n))), 'f').lstrip('.0').rstrip('0.'))
   return initial_sign + str(x)+'_'+str(int(round(abs(np.log10(n/x)))))

def load_fixation_data(type_map, string_saving, repetitions, load_run_details=False):
   discovery_time_array = np.load('./evolution_data/'+type_map+'_fixation_discovery_time_allruns_array'+string_saving+'.npy' )
   fixation_time_array = np.load('./evolution_data/'+type_map+'_fixation_time_allruns_array'+string_saving+'.npy' )
   fixating_pheno_array = np.load('./evolution_data/'+type_map+'_fixating_pheno_array'+string_saving+'.npy' )
   run_vs_times_appeared_p1, run_vs_times_appeared_p2, run_vs_genotype_list = {}, {}, {}
   if load_run_details:
      for run in range(repetitions):
         run_vs_times_appeared_p1[run] = np.load('./evolution_data/'+type_map+'_fixation_times_new_p1_'+string_saving+'run'+str(run)+'.npy' )
         run_vs_times_appeared_p2[run] = np.load('./evolution_data/'+type_map+'_fixation_times_new_p2_'+string_saving+'run'+str(run)+'.npy')
         try:
            run_vs_genotype_list[run] = np.load('./evolution_data/'+type_map+'_fixation_genotype_path_'+string_saving+'run'+str(run)+'.npy' )
         except IOError:
           pass
   return discovery_time_array, fixation_time_array, fixating_pheno_array, run_vs_times_appeared_p1, run_vs_times_appeared_p2, run_vs_genotype_list

def roundT_up(T):
    exp = int(np.floor(np.log10(T)))
    if T > T//10**exp * 10**exp:
       a = T//10**exp + 1
    else:
        a = T//10**exp
    return  int(a * 10**exp)



############################################################################################################
## test
############################################################################################################
if __name__ == "__main__":
   assert format_integer_filename(8) == '8_0'
   assert format_integer_filename(-8) == 'minus8_0'
   assert format_integer_filename(-80) == 'minus8_1'
   assert format_integer_filename(8000) == '8_3'
   assert format_integer_filename(8800) == '88_2'
   ###
   assert format_float_filename(0.88) == '88_2'
   assert format_float_filename(-0.88) == 'minus88_2'
   assert format_float_filename(-0.8) == 'minus8_1'
   assert format_float_filename(-0.812) == 'minus812_3'
   assert format_float_filename(-0.0812) == 'minus812_4'
   assert format_float_filename(-0.0002) == 'minus2_4'
   ###
   assert roundT_up(355) == 400
   assert roundT_up(2100) == 3000
   assert roundT_up(3300) == 4000

   print('finished tests')
   

