#!/bin/python3
import timeit
import pandas as pd
import numpy as np 

start = timeit.default_timer()

pivot_table = pd.read_csv('../data/timeseries.csv', sep=';', low_memory=False)

pivot_table.iloc[1:1:] = pivot_table.iloc[1:1:].replace(',', '.', regex=True)#.astype(float)

def convert_and_truncate(value, decimal_places=2):
    try:
        # Substituir ',' por '.' e, em seguida, converter para float
        float_value = float(value.replace(',', '.'))
        truncated_value = float(f"{float_value:.{decimal_places}f}")
        return truncated_value
    except ValueError:
        return None

pivot_table[1:,1:] = pivot_table[1:,1:].apply(convert_and_truncate)

# print('Head:', pivot_table.head())
# print(pivot_table.iloc[0])
# print('Colunas:', pivot_table.columns)
# print('Index:', pivot_table.index)

print(pivot_table['Part'].head())

end = timeit.default_timer()
exec_time = end - start
print(f'Código executado em {exec_time} minutos')
