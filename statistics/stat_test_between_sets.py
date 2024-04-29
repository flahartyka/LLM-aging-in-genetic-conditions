import os,sys,re
import numpy as np 
import pickle 
import pandas as pd 
from scipy.stats import mannwhitneyu
from scipy import stats
import statistics
from mlxtend.evaluate import permutation_test


workdir = '/data/flahartyka/LLM-age-related-genetic-data'
fin = '70b_vignettes_scoring.csv'
fin1 = 'GPT_vignettes_scoring.csv'
fin2 = 'managementGPT.csv'


df = pd.read_csv(os.path.join(workdir,fin2),encoding='latin1') # https://stackoverflow.com/questions/71419895/utf-8-codec-cant-decode-byte-0xed

# 
df = df.dropna(subset=["Correctness"]) # ! drop things we not grade yet
df = df.reset_index(drop=True)

#
df['Category'] = [s.strip() for s in df['Category'].tolist() ]

set ( df['Category'].to_list() ) 

categories_total = ['Only childhood', 'Only adulthood', 'Change: presentation/management', 'No change', 'Change in presentation related to age']


# ! convert to integer. 
for key_col_name in ['Correctness', 'Conciseness', 'Completeness', 'Compassion', 'Total', 'Accuracy']: 
  for this_name in [x for x in df.columns if key_col_name in x]:
    df[this_name] = [ float (x) for x in df[this_name].tolist() ] # if str(x).isnumeric() else np.nan 


# ---------------------------------------------------------------------------- #
category1 = []
metric1 = []
mean1 = []
stdv1 = []
category2 = []
metric2 = []
mean2 = []
stdv2 = []
pval = []


def ttest_unpair_set ( test_setting, category_type_1, category_type_2, metric_1, metric_2 ): 
  
  temp_df1 = df[ df['Category'] == category_type_1 ] # 'Only childhood' ]
  temp_df2 = df[ df['Category'] == category_type_2 ] # 
  
  x = temp_df1[metric_1].tolist() # 'Correctness.1'
  y = temp_df2[metric_2].tolist() 
  
  if test_setting == 'unpair':
    t_stat, p_val = stats.ttest_ind(x, y, equal_var=False, nan_policy='omit')  
  
  if test_setting == 'pair':
    t_stat, p_val = stats.ttest_rel(x, y, nan_policy='omit') # ! some version return more than t_stat, p_val, we may need to change this if needed. 
  cleanedList = [i for i in x if str(i) != 'nan']
  cleanedList1 = [j for j in y if str(j) != 'nan']
  
  # ! print or save some where? 
  
  print('\nmean 1', category_type_1, metric_1, str(np.nanmean(x)))
  category1.append(category_type_1)
  metric1.append(metric_1)
  mean1.append(np.nanmean(x))
  stdv1.append(statistics.stdev(cleanedList))
  category2.append(category_type_2)
  metric2.append(metric_2)
  mean2.append(np.nanmean(y))
  stdv2.append(statistics.stdev(cleanedList1))
  pval.append(p_val)
  print('stdev:', str(statistics.stdev(cleanedList)))
  print ('usable sample', len(x)) # ! may have some NaN removed from "?0"
  print('mean 2', category_type_2, metric_2, str(np.nanmean(y)))
  print('stdev:', str(statistics.stdev(cleanedList1)))
  print ('usable sample', len(y))
  print("t-statistic = " + str(t_stat))  
  print("p-value = " + str(p_val),'\n')


# ---------------------------------------------------------------------------- #

# ! number of things to test. 
# ! pair t-test or not, disease category based on age, metric. 

for category in categories_total:
  things_to_test = dict()
  things_to_test = {  1: [ 'pair', category, category, 'Correctness', 'Correctness.1' ] ,
                    2: [ 'pair', category, category, 'Completeness', 'Completeness.1'] ,
                    3: ['pair', category, category, 'Conciseness & Clarity', 'Conciseness & Clarity.1'],
                    4: ['pair', category, category, 'Total (0-3)', 'Total (0-3).1'],
                    5: ['pair', category, category, 'Total (0-2)', 'Total (0-2).1'],
                    6: ['pair', category, category, 'Accuracy', 'Accuracy.1'],
                    7: [ 'pair', category, category, 'Correctness', 'Correctness.2' ] ,
                    8: [ 'pair', category, category, 'Completeness', 'Completeness.2'] ,
                    9: ['pair', category, category, 'Conciseness & Clarity', 'Conciseness & Clarity.2'],
                    10: ['pair', category, category, 'Total (0-3)', 'Total (0-3).2'],
                    11: ['pair', category, category, 'Total (0-2)', 'Total (0-2).2'],
                    12: ['pair', category, category, 'Accuracy', 'Accuracy.2'],
                    13: [ 'pair', category, category, 'Correctness.2', 'Correctness.3' ] ,
                    14: [ 'pair', category, category, 'Completeness.2', 'Completeness.3'] ,
                    15: ['pair', category, category, 'Conciseness & Clarity.2', 'Conciseness & Clarity.3'],
                    16: ['pair', category, category, 'Total (0-3).2', 'Total (0-3).3'],
                    17: ['pair', category, category, 'Total (0-2).2', 'Total (0-2).3'],
                    18: ['pair', category, category, 'Accuracy.2', 'Accuracy.3'],
                    19: [ 'pair', category, category, 'Correctness.1', 'Correctness.3' ] ,
                    20: [ 'pair', category, category, 'Completeness.1', 'Completeness.3'] ,
                    21: ['pair', category, category, 'Conciseness & Clarity.1', 'Conciseness & Clarity.3'],
                    22: ['pair', category, category, 'Total (0-3).1', 'Total (0-3).3'],
                    23: ['pair', category, category, 'Total (0-2).1', 'Total (0-2).3'],
                    24: ['pair', category, category, 'Accuracy.1', 'Accuracy.3'],
                    }
  ttest_unpair_set ( *things_to_test[1] ) 
  ttest_unpair_set ( *things_to_test[2] ) 
  ttest_unpair_set ( *things_to_test[3] ) 
  ttest_unpair_set ( *things_to_test[4] ) 
  ttest_unpair_set ( *things_to_test[5] ) 
  ttest_unpair_set ( *things_to_test[6] ) 
  ttest_unpair_set ( *things_to_test[7] ) 
  ttest_unpair_set ( *things_to_test[8] ) 
  ttest_unpair_set ( *things_to_test[9] ) 
  ttest_unpair_set ( *things_to_test[10] ) 
  ttest_unpair_set ( *things_to_test[11] ) 
  ttest_unpair_set ( *things_to_test[12] ) 
  ttest_unpair_set ( *things_to_test[13] ) 
  ttest_unpair_set ( *things_to_test[14] ) 
  ttest_unpair_set ( *things_to_test[15] )
  ttest_unpair_set ( *things_to_test[16] )  
  ttest_unpair_set ( *things_to_test[17] )  
  ttest_unpair_set ( *things_to_test[18] )  
  ttest_unpair_set ( *things_to_test[19] )  
  ttest_unpair_set ( *things_to_test[20] )  
  ttest_unpair_set ( *things_to_test[21] )  
  ttest_unpair_set ( *things_to_test[22] )  
  ttest_unpair_set ( *things_to_test[23] )  
  ttest_unpair_set ( *things_to_test[24] )  


dict1 = {'category1': category1,'metric1': metric1,'mean1': mean1,'stdev1':stdv1,'category2': category2, 'metric2': metric2,'mean2': mean2,'stdev2':stdv2,'p-value': pval}

results = pd.DataFrame(dict1)
results.to_csv('70b_pvals.csv')

# ---------------------------------------------------------------------------- #

fin1 = 'GPT_vignettes_scoring.csv'

df = pd.read_csv(os.path.join(workdir,fin1),encoding='latin1') # https://stackoverflow.com/questions/71419895/utf-8-codec-cant-decode-byte-0xed
#df1 = pd.read_csv(os.path.join(workdir,fin1),encoding='latin1') # https://stackoverflow.com/questions/71419895/utf-8-codec-cant-decode-byte-0xed

# 
df = df.dropna(subset=["Correctness"]) # ! drop things we not grade yet
df = df.reset_index(drop=True)

df['Category'] = [s.strip() for s in df['Category'].tolist() ]

set ( df['Category'].to_list() ) 

categories_total = ['Only childhood', 'Only adulthood', 'Change: presentation/management', 'No change', 'Change in presentation related to age']

for key_col_name in ['Correctness', 'Conciseness', 'Completeness', 'Compassion' 'Total',]: 
  for this_name in [x for x in df.columns if key_col_name in x]:
    df[this_name] = [ float (x) for x in df[this_name].tolist() ] # if str(x).isnumeric() else np.nan 


category1 = []
metric1 = []
mean1 = []
stdv1 = []
category2 = []
metric2 = []
mean2 = []
stdv2 = []
pval = []


for category in categories_total:
  things_to_test = dict()
  things_to_test = {  1: [ 'pair', category, category, 'Correctness', 'Correctness.1' ] ,
                    2: [ 'pair', category, category, 'Completeness', 'Completeness.1'] ,
                    3: ['pair', category, category, 'Conciseness & Clarity', 'Conciseness & Clarity.1'],
                    4: ['pair', category, category, 'Total (0-3)', 'Total (0-3).1'],
                    5: ['pair', category, category, 'Total (0-2)', 'Total (0-2).1'],
                    6: ['pair', category, category, 'Accuracy', 'Accuracy.1'],
                    }
  ttest_unpair_set ( *things_to_test[1] ) 
  ttest_unpair_set ( *things_to_test[2] ) 
  ttest_unpair_set ( *things_to_test[3] ) 
  ttest_unpair_set ( *things_to_test[4] )  
  ttest_unpair_set ( *things_to_test[5] )  
  ttest_unpair_set ( *things_to_test[6] )  


dict1 = {'category1': category1,'metric1': metric1,'mean1': mean1,'stdev1':stdv1,'category2': category2, 'metric2': metric2,'mean2': mean2,'stdev2':stdv2,'p-value': pval}

results = pd.DataFrame(dict1)
results.to_csv('GPT_pvals.csv')


# ---------------------------------------------------------------------------- #
#Compare 70b to GPT
fin = '70b_vignettes_scoring.csv'
fin1 = 'GPT_vignettes_scoring.csv'


df = pd.read_csv(os.path.join(workdir,fin),encoding='latin1') # https://stackoverflow.com/questions/71419895/utf-8-codec-cant-decode-byte-0xed
df1 = pd.read_csv(os.path.join(workdir,fin1),encoding='latin1') # https://stackoverflow.com/questions/71419895/utf-8-codec-cant-decode-byte-0xed

# 
df = df.dropna(subset=["Correctness"]) # ! drop things we not grade yet
df = df.reset_index(drop=True)

df1 = df1.dropna(subset=["Correctness"]) # ! drop things we not grade yet
df1 = df1.reset_index(drop=True)


#
df['Category'] = [s.strip() for s in df['Category'].tolist() ]
df1['Category'] = [s.strip() for s in df1['Category'].tolist() ]
set ( df['Category'].to_list() ) 
set ( df1['Category'].to_list() ) 

categories_total = ['Only childhood', 'Only adulthood', 'Change: presentation/management', 'No change', 'Change in presentation related to age']


# ! convert to integer. 
for key_col_name in ['Correctness', 'Conciseness', 'Completeness', 'Compassion' 'Total',]: 
  for this_name in [x for x in df.columns if key_col_name in x]:
    df[this_name] = [ float (x) for x in df[this_name].tolist() ] # if str(x).isnumeric() else np.nan 

for key_col_name in ['Correctness', 'Conciseness', 'Completeness', 'Compassion' 'Total',]: 
  for this_name in [x for x in df1.columns if key_col_name in x]:
    df1[this_name] = [ float (x) for x in df1[this_name].tolist() ] # if str(x).isnumeric() else np.nan 

model1 = []
model2 = []
category1 = []
metric1 = []
mean1 = []
stdv1 = []
category2 = []
metric2 = []
mean2 = []
stdv2 = []
pval = []


def ttest_unpair_set ( test_setting, category_type_1, category_type_2, metric_1, metric_2 ): 
  
  temp_df1 = df [ df['Category'] == category_type_1 ] # 'Only childhood' ]
  temp_df2 = df1 [ df1['Category'] == category_type_2 ] # 
  
  x = temp_df1[metric_1].tolist() # 'Correctness.1'
  y = temp_df2[metric_2].tolist() 
  
  if test_setting == 'unpair':
    t_stat, p_val = stats.ttest_ind(x, y, equal_var=False, nan_policy='omit')  
  
  if test_setting == 'pair':
    t_stat, p_val = stats.ttest_rel(x, y, nan_policy='omit') # ! some version return more than t_stat, p_val, we may need to change this if needed. 
  cleanedList = [i for i in x if str(i) != 'nan']
  cleanedList1 = [j for j in y if str(j) != 'nan']
  
  # ! print and save
  
  print('\nmean 1', category_type_1, metric_1, str(np.nanmean(x)))
  model1.append('70b')
  model2.append('GPT')
  category1.append(category_type_1)
  metric1.append(metric_1)
  mean1.append(np.nanmean(x))
  stdv1.append(statistics.stdev(cleanedList))
  category2.append(category_type_2)
  metric2.append(metric_2)
  mean2.append(np.nanmean(y))
  stdv2.append(statistics.stdev(cleanedList1))
  pval.append(p_val)
  print('stdev:', str(statistics.stdev(cleanedList)))
  print ('usable sample', len(x)) # ! may have some NaN removed from "?0"
  print('mean 2', category_type_2, metric_2, str(np.nanmean(y)))
  print('stdev:', str(statistics.stdev(cleanedList1)))
  print ('usable sample', len(y))
  print("t-statistic = " + str(t_stat))  
  print("p-value = " + str(p_val),'\n')




for category in categories_total:
  things_to_test = dict()
  things_to_test = {  1: [ 'unpair', category, category, 'Correctness', 'Correctness' ] ,
                    2: [ 'unpair', category, category, 'Completeness', 'Completeness'] ,
                    3: ['unpair', category, category, 'Conciseness & Clarity', 'Conciseness & Clarity'],
                    4: ['unpair', category, category, 'Total (0-3)', 'Total (0-3)'],
                    5: ['unpair', category, category, 'Total (0-2)', 'Total (0-2)'],
                    6: ['unpair', category, category, 'Accuracy', 'Accuracy'],
                    7: [ 'unpair', category, category, 'Correctness.2', 'Correctness' ] ,
                    8: [ 'unpair', category, category, 'Completeness.2', 'Completeness'] ,
                    9: ['unpair', category, category, 'Conciseness & Clarity.2', 'Conciseness & Clarity'],
                    10: ['unpair', category, category, 'Total (0-3).2', 'Total (0-3)'],
                    11: ['unpair', category, category, 'Total (0-2).2', 'Total (0-2)'],
                    12: ['unpair', category, category, 'Accuracy.2', 'Accuracy'],
                    13: [ 'unpair', category, category, 'Correctness.1', 'Correctness.1' ] ,
                    14: [ 'unpair', category, category, 'Completeness.1', 'Completeness.1'] ,
                    15: ['unpair', category, category, 'Conciseness & Clarity.1', 'Conciseness & Clarity.1'],
                    16: ['unpair', category, category, 'Total (0-3).1', 'Total (0-3).1'],
                    17: ['unpair', category, category, 'Total (0-2).1', 'Total (0-2).1'],
                    18: ['unpair', category, category, 'Accuracy.1', 'Accuracy.1'],
                    19: [ 'unpair', category, category, 'Correctness.3', 'Correctness.1' ] ,
                    20: [ 'unpair', category, category, 'Completeness.3', 'Completeness.1'] ,
                    21: ['unpair', category, category, 'Conciseness & Clarity.3', 'Conciseness & Clarity.1'],
                    22: ['unpair', category, category, 'Total (0-3).3', 'Total (0-3).1'],
                    23: ['unpair', category, category, 'Total (0-2).3', 'Total (0-2).1'],
                    24: ['unpair', category, category, 'Accuracy.3', 'Accuracy.1'],
                    }
  ttest_unpair_set ( *things_to_test[1] ) 
  ttest_unpair_set ( *things_to_test[2] ) 
  ttest_unpair_set ( *things_to_test[3] ) 
  ttest_unpair_set ( *things_to_test[4] ) 
  ttest_unpair_set ( *things_to_test[5] ) 
  ttest_unpair_set ( *things_to_test[6] ) 
  ttest_unpair_set ( *things_to_test[7] ) 
  ttest_unpair_set ( *things_to_test[8] ) 
  ttest_unpair_set ( *things_to_test[9] ) 
  ttest_unpair_set ( *things_to_test[10] ) 
  ttest_unpair_set ( *things_to_test[11] ) 
  ttest_unpair_set ( *things_to_test[12] ) 
  ttest_unpair_set ( *things_to_test[13] ) 
  ttest_unpair_set ( *things_to_test[14] ) 
  ttest_unpair_set ( *things_to_test[15] )
  ttest_unpair_set ( *things_to_test[16] )  
  ttest_unpair_set ( *things_to_test[17] ) 
  ttest_unpair_set ( *things_to_test[18] ) 
  ttest_unpair_set ( *things_to_test[19] ) 
  ttest_unpair_set ( *things_to_test[20] ) 
  ttest_unpair_set ( *things_to_test[21] ) 
  ttest_unpair_set ( *things_to_test[22] ) 
  ttest_unpair_set ( *things_to_test[23] ) 
  ttest_unpair_set ( *things_to_test[24] ) 


dict1 = {'model1': model1, 'category1': category1,'metric1': metric1,'mean1': mean1,'stdev1':stdv1, 'model2': model2, 'category2': category2, 'metric2': metric2,'mean2': mean2,'stdev2':stdv2,'p-value': pval}

results = pd.DataFrame(dict1)
results.to_csv('both_models_pvals.csv')



