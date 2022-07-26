import numpy as np
import pandas as pd

order = pd.read_csv('space_4_time_1.csv', sep=' ')
print('=== Space 1 Time 1 ===')
print("N\tAbsolute\tRelative")
for i in range(order.shape[0] - 1):
    abs_order = np.log2(order['l2_abs'][i] / order['l2_abs'][i+1])
    rel_order = np.log2(order['l2_rel'][i] / order['l2_rel'][i+1])
    n = order['n'][i]
    print(f"{n}\t{abs_order:.3f}\t\t{rel_order:.3f}")

# order = pd.read_csv('space_1_time_2.csv', sep=' ')
# print('\n=== Space 1 Time 2 ===')
# print("Absolute\tRelative")
# for i in range(order.shape[0] - 1):
#     abs_order = np.log2(order['l2_abs'][i] / order['l2_abs'][i+1])
#     rel_order = np.log2(order['l2_rel'][i] / order['l2_rel'][i+1])
#     print(f'{abs_order:.3f}\t\t{rel_order:.3f}')

order = pd.read_csv('space_4_time_2.csv', sep=' ')
print('\n=== Space 2 Time 2 ===')
print("N\tAbsolute\tRelative")
for i in range(order.shape[0] - 1):
    abs_order = np.log2(order['l2_abs'][i] / order['l2_abs'][i+1])
    rel_order = np.log2(order['l2_rel'][i] / order['l2_rel'][i+1])
    n = order['n'][i]
    print(f"{n}\t{abs_order:.3f}\t\t{rel_order:.3f}")

# order = pd.read_csv('space_3_time_1.csv', sep=' ')
# print('\n=== Space 3 Time 1 ===')
# print("Absolute\tRelative")
# for i in range(order.shape[0] - 1):
#     abs_order = np.log2(order['l2_abs'][i] / order['l2_abs'][i+1])
#     rel_order = np.log2(order['l2_rel'][i] / order['l2_rel'][i+1])
#     print(f'{abs_order:.3f}\t\t{rel_order:.3f}')

# order = pd.read_csv('space_3_time_2.csv', sep=' ')
# print('\n=== Space 3 Time 2 ===')
# print("Absolute\tRelative")
# for i in range(order.shape[0] - 1):
#     abs_order = np.log2(order['l2_abs'][i] / order['l2_abs'][i+1])
#     rel_order = np.log2(order['l2_rel'][i] / order['l2_rel'][i+1])
#     print(f'{abs_order:.3f}\t\t{rel_order:.3f}')

order = pd.read_csv('space_4_time_3.csv', sep=' ')
print('\n=== Space 3 Time 3 ===')
print("N\tAbsolute\tRelative")
for i in range(order.shape[0] - 1):
    abs_order = np.log2(order['l2_abs'][i] / order['l2_abs'][i+1])
    rel_order = np.log2(order['l2_rel'][i] / order['l2_rel'][i+1])
    n = order['n'][i]
    print(f"{n}\t{abs_order:.3f}\t\t{rel_order:.3f}")

# order = pd.read_csv('space_4_time_4.csv', sep=' ')
# print('\n=== Space 4 Time 4 ===')
# print("N\tAbsolute\tRelative")
# for i in range(order.shape[0] - 1):
#     abs_order = np.log2(order['l2_abs'][i] / order['l2_abs'][i+1])
#     rel_order = np.log2(order['l2_rel'][i] / order['l2_rel'][i+1])
#     n = order['n'][i]
#     print(f"{n}\t{abs_order:.3f}\t\t{rel_order:.3f}")
