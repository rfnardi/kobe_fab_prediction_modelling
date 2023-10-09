#!/bin/python3
import timeit
import pandas as pd
import numpy as np 

start = timeit.default_timer()

df = pd.read_csv('../data/timeseries.csv', sep=';')

print('Head:', df.head())
print(df.iloc[0])
print('Colunas:', df.columns)
print('Index:', df.index)

end = timeit.default_timer()
exec_time = end - start
print(f'Código executado em {exec_time} minutos')
