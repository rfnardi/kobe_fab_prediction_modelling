#!/bin/python3 

import timeit

start = timeit.default_timer()

import sys
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
from keras.regularizers import l2, l1

pivot_table = pd.read_csv('../../data/timeseries_ready_to_go.csv', low_memory=False)


# pd.pivot = pivot_table.transpose()

# print(pivot_table['Unnamed: 0'])

pivot_table.drop(columns=['Unnamed: 0'], inplace=True)

# pivot_table = pivot_table.replace(r',','').astype(float)
# pivot_table = pivot_table.applymap(lambda x: str(x).replace(',', ''))
# pivot_table = pivot_table.applymap(lambda x: str(x).replace('0.0', '0'))
# pivot_table = pivot_table.applymap(lambda x: str(x).replace('.', ''))
# pivot_table = pivot_table.applymap(lambda x: str(x).replace('-', ''))

pivot_table = pivot_table.astype(float)

min_ = pivot_table.min().min()
max_ = pivot_table.max().max()

print(f'Mínimo: {min_} e máximo {max_}:')

print(pivot_table.head())

# exit(1)

print('---------------Colunas:')
print(pivot_table.columns[:5])

n_input_series = len(pivot_table.columns)
n_output_series = 10
# sorted_series = pd.DataFrame(pivot_table).sum().sort_values(ascending=False)

sums = []
cols = []
for col in pivot_table.columns:
    sum_ = 0
    sum_ = pivot_table[col].sum()
    sums.append(sum_)
    cols.append(col)

sum_df = pd.DataFrame({'cols' : cols, 'sums' : sums})
# sum_df['sums'] = sum_df['sums'].replace(',', '.').astype(float)

# arrumar os dados!!!!! Não está conseguindo passar pra floats
# sum_df['sums'] = sum_df['sums']

print('sum_df info:')
print(sum_df.info())


# print(sum_df.head())

ordered_sums = sum_df.sort_values(by='sums')#.reset_index(drop=True)


print('---------------------')
print('ordered_sums:')
print(ordered_sums.head())
print('---------------------')

ordered_pivot_table = pivot_table[ordered_sums['cols']]

ordered_transposed_pt = ordered_pivot_table.transpose()


selected_series = ordered_transposed_pt.head(n_output_series)
# selected_series = selected_series.transpose()

output_series = pivot_table[selected_series]

# ------------------ Splitting data into train + validation + test:
train_size = int(len(pivot_table) * 0.8)
train_plus_val_size = int(len(pivot_table) * 0.9)

train_table_X = pivot_table[:train_size]
train_table_Y = output_series[:train_size]

val_table_X = pivot_table[train_size:train_plus_val_size]
val_table_Y = output_series[train_size:train_plus_val_size]

test_table_X = pivot_table[train_plus_val_size:]
test_table_Y = output_series[train_plus_val_size:]
# ----------------- End Splitting

def Data_to_X_Y(X_table, Y_table, n_lags, horizon):
    X = []
    for item in X_table.columns:
        x = []
        serie_as_np = np.array(X_table[item])
        for i in range(len(serie_as_np) - n_lags - horizon ):
            row_X = [a for a in serie_as_np[i:i+n_lags]]
            x.append(row_X)
        X.append(x)

    Y = []
    for item in Y_table.columns:
        y = []
        serie_as_np = np.array(Y_table[item])
        for i in range(len(serie_as_np) - n_lags - horizon ):
            row_Y = [a for a in serie_as_np[i+n_lags:i+n_lags+horizon]]
            y.append(row_Y)
        Y.append(y)

    X, Y = np.array(X) , np.array(Y)
    X = X.transpose(0,2,1)
    Y = Y.transpose(0,2,1)
    return X, Y


X_train, Y_train = Data_to_X_Y(train_table_X, train_table_Y, 90, 7)

print('X_train.shape:', X_train.shape)
print('Y_train.shape:', Y_train.shape)


end = timeit.default_timer()

exec_time = round(end - start, 2) 

print(f'Tempo de execução: {exec_time}')
