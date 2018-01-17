import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns


font = {'family' : 'normal',
        'weight' : 'light',
        'size'   : 10}

susc = 1.20866
heatc = 1.14904
mag = 0.776831

matplotlib.rc('font', **font)

sns.set(style='ticks', palette='Set2')
palette = sns.color_palette()

path1 = "./sgd_mixed3.txt"
path2 = "./sgd_mixed6.txt"

names = ['Epochs', 'Magnetisation', 'Susceptibility', 'Heat Capacity']

df1 = pd.read_csv(path1, sep="\t", header=0, names=names)
df2 = pd.read_csv(path2, sep="\t", header=0, names=names)

df1.Susceptibility = df1.Susceptibility
df2.Susceptibility = df2.Susceptibility

#df1_err = df1.rolling(6, min_periods=1).std()
#df2_err = df2.rolling(6, min_periods=1).std()

#df1 = df1.rolling(10, min_periods=1).mean()
#df2 = df2.rolling(6, min_periods=1).mean()

plt.figure(figsize=(6.0, 3.5))

plt.plot(df1.Epochs, df1.Susceptibility, linewidth=1)
#plt.fill_between(df1.Epochs, df1.Susceptibility - df1_err.Susceptibility, df1.Susceptibility + df1_err.Susceptibility, alpha=0.2)

plt.plot(df2.Epochs, df2.Susceptibility, linewidth=1)
#plt.fill_between(df2.Epochs, df2.Susceptibility - df2_err.Susceptibility, df2.Susceptibility + df2_err.Susceptibility, alpha=0.2)

plt.axhline(susc, ls=':', linewidth=1.5, color=palette[3])

#plt.legend(['Adadelta', 'SGD, lr = 0.01'], loc=2)

plt.xlabel(names[0])
plt.ylabel(names[2])

plt.tight_layout()

plt.show()