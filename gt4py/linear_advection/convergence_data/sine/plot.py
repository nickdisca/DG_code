import pandas as pd
import matplotlib.pyplot as plt

for i in range(1, 5):
    filename = f'space_{i}_time_{i}.csv'
    df = pd.read_csv(filename, sep=' ')
    plt.plot(df['h'][:4], df['l2_abs'][:4], marker='.')

plt.xscale('log')
plt.yscale('log')
plt.show()
