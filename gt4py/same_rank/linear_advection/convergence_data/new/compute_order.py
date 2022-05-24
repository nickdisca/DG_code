import numpy as np
import pandas as pd

order1 = pd.read_csv('space_1_time_1.csv', sep=' ')
print('=== Order 1: ===')
print("Absolute\t\tRelative")
for i in range(order1.shape[0] - 1):
    abs_order = np.log2(order1['l2_abs'][i] / order1['l2_abs'][i+1])
    rel_order = np.log2(order1['l2_rel'][i] / order1['l2_rel'][i+1])
    print(f'{abs_order}   {rel_order}')

order2 = pd.read_csv('space_2_time_2.csv', sep=' ')
print('\n=== Order 2: ===')
print("Absolute\t\tRelative")
for i in range(order2.shape[0] - 1):
    abs_order = np.log2(order2['l2_abs'][i] / order2['l2_abs'][i+1])
    rel_order = np.log2(order2['l2_rel'][i] / order2['l2_rel'][i+1])
    print(f'{abs_order}   {rel_order}')

order3 = pd.read_csv('space_3_time_3.csv', sep=' ')
print('\n=== Order 3: ===')
print("Absolute\t\tRelative")
for i in range(order3.shape[0] - 1):
    abs_order = np.log2(order3['l2_abs'][i] / order3['l2_abs'][i+1])
    rel_order = np.log2(order3['l2_rel'][i] / order3['l2_rel'][i+1])
    print(f'{abs_order}   {rel_order}')
