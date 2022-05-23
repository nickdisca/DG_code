import numpy as np
import pandas as pd

order1 = pd.read_csv('space_1_time_1.dat', sep=' ')
order2 = pd.read_csv('space_2_time_2.dat', sep=' ')
order3 = pd.read_csv('space_3_time_3.dat', sep=' ')

print('=== Order 1: ===')
for i in range(order1.shape[0] - 1):
    print(np.log2(order1['l2_error'][i] / order1['l2_error'][i+1]))

print('\n=== Order 2: ===')
for i in range(order2.shape[0] - 1):
    print(np.log2(order2['l2_error'][i] / order2['l2_error'][i+1]))

print('\n=== Order 3: ===')
for i in range(order3.shape[0] - 1):
    print(np.log2(order3['l2_error'][i] / order3['l2_error'][i+1]))
