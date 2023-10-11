#!/bin/python3
import timeit
import pandas as pd
import numpy as np 

start = timeit.default_timer()

pivot_table = pd.read_csv('../data/timeseries_original.csv', sep=';', low_memory=False)

pivot_table.iloc[1:,1:] = pivot_table.iloc[1:,1:].replace(',', '.', regex=True).astype(float)

pivot_table.to_csv('../data/timeseries_comma_fixed.csv')

# print('Head:', pivot_table.head())
# print(pivot_table.iloc[0])
# print('Colunas:', pivot_table.columns)
# print('Index:', pivot_table.index)

# print(pivot_table.index)

# print(pivot_table['Part'])

end = timeit.default_timer()
exec_time = end - start
print(f'Código executado em {exec_time} minutos')
