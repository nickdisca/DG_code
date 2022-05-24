import numpy as np
import pandas as pd

order = pd.read_csv('space_1_time_1.csv', sep=' ')
print('=== Space 1 Time 1 ===')
print("Absolute\t\tRelative")
for i in range(order.shape[0] - 1):
    abs_order = np.log2(order['l2_abs'][i] / order['l2_abs'][i+1])
    rel_order = np.log2(order['l2_rel'][i] / order['l2_rel'][i+1])
    print(f'{abs_order}   {rel_order}')

order = pd.read_csv('space_1_time_2.csv', sep=' ')
print('\n=== Space 1 Time 2 ===')
print("Absolute\t\tRelative")
for i in range(order.shape[0] - 1):
    abs_order = np.log2(order['l2_abs'][i] / order['l2_abs'][i+1])
    rel_order = np.log2(order['l2_rel'][i] / order['l2_rel'][i+1])
    print(f'{abs_order}   {rel_order}')

order = pd.read_csv('space_2_time_2.csv', sep=' ')
print('\n=== Space 2 Time 2 ===')
print("Absolute\t\tRelative")
for i in range(order.shape[0] - 1):
    abs_order = np.log2(order['l2_abs'][i] / order['l2_abs'][i+1])
    rel_order = np.log2(order['l2_rel'][i] / order['l2_rel'][i+1])
    print(f'{abs_order}   {rel_order}')

order = pd.read_csv('space_3_time_1.csv', sep=' ')
print('\n=== Space 3 Time 1 ===')
print("Absolute\t\tRelative")
for i in range(order.shape[0] - 1):
    abs_order = np.log2(order['l2_abs'][i] / order['l2_abs'][i+1])
    rel_order = np.log2(order['l2_rel'][i] / order['l2_rel'][i+1])
    print(f'{abs_order}   {rel_order}')

order = pd.read_csv('space_3_time_2.csv', sep=' ')
print('\n=== Space 3 Time 2 ===')
print("Absolute\t\tRelative")
for i in range(order.shape[0] - 1):
    abs_order = np.log2(order['l2_abs'][i] / order['l2_abs'][i+1])
    rel_order = np.log2(order['l2_rel'][i] / order['l2_rel'][i+1])
    print(f'{abs_order}   {rel_order}')

order = pd.read_csv('space_3_time_3.csv', sep=' ')
print('\n=== Space 3 Time 3 ===')
print("Absolute\t\tRelative")
for i in range(order.shape[0] - 1):
    abs_order = np.log2(order['l2_abs'][i] / order['l2_abs'][i+1])
    rel_order = np.log2(order['l2_rel'][i] / order['l2_rel'][i+1])
    print(f'{abs_order}   {rel_order}')
