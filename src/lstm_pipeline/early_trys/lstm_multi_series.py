#!/bin/python3

import timeit
from datetime import datetime
from scipy.spatial import transform
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
start = timeit.default_timer()
import pandas as pd 
import numpy as np 
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
from statsmodels.graphics.tsaplots import pacf, acf, plot_acf, plot_pacf
from keras.regularizers import l2, l1

pivot_table = pd.read_csv('../data/timeseries_comma_fixed.csv')

pivot_table = pivot_table.transpose()

sorted_serie = pivot_table.sum().sort_values(ascending=False)

n_output_series = 100
top_sales = sorted_serie.head(n_output_series)

top_sales_items = top_sales.index

top_series = pivot_table[top_sales_items]

top_series.shape

scaler = MinMaxScaler()
pivot_table_renormalized = pd.DataFrame(scaler.fit_transform(top_series.iloc[1:,1:]), columns=top_sales_items)

# <h2 style="color:black; background-color: gray; font-weight: bold">Preparando dados para treino</h2>

# Divisão em conjuntos de treinamento, validação e teste

train_size = int(len(pivot_table_renormalized) * 0.8)
train_plus_val_size = int(len(pivot_table_renormalized) * 0.9)
train_data = pivot_table_renormalized[:train_size]
val_data = pivot_table_renormalized[train_size:train_plus_val_size]
test_data = pivot_table_renormalized[train_plus_val_size:]

# Preparação dos dados de treinamento, validação e teste
# Vamos usar os dados dos últimos N dias para prever os próximos 28

def Data_to_X_Y(P_table, N):
    X = []
    Y = []
    for item in P_table.columns:
        x = []
        y = []
        serie_as_np = np.array(P_table[item])
        for i in range(len(serie_as_np) - N - 28 ):
            row_X = [a for a in serie_as_np[i:i+N]]
            row_Y = [a for a in serie_as_np[i+N:i+N+28]]
            x.append(row_X)
            y.append(row_Y)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X) , np.array(Y)
    X = X.transpose(0,2,1)
    Y = Y.transpose(0,2,1)
    return X, Y

N_lags = 70 #usar múltimplos de 7 para que o reshape funcione bem
forecast_horizon = 28
n_input_series = 100
X_train, Y_train = Data_to_X_Y(train_data, N_lags)

print('X_train:', X_train.shape)
print('Y_train:', Y_train.shape)

# X_train: (1472, 30, 100)
# Y_train: (1472, 28, 100)

print('Check 4')

# <h2 style="color:black; background-color: gray; font-weight: bold">Modelo:</h2>

# ***************************************** Arquitetura do Modelo:

_lambda = 0.01

model = Sequential()
model.add(InputLayer((N_lags, n_input_series)))
# model.add(LSTM(80, kernel_regularizer=l1(_lambda), return_sequences=True))

model.add(LSTM(80, kernel_regularizer=l1(_lambda), return_sequences=True))
model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(50, activation='relu'))

last_layer_size = forecast_horizon*n_output_series

# model.add(Dropout(0.2))
model.add(Dense(last_layer_size, activation='linear')) 

model.add(Reshape((forecast_horizon,n_input_series)))

model.summary()

# ******************************************************************

# salvar o melhor modelo:
# check_point = ModelCheckpoint('../../models/task3/', save_best_only=True)

print('Check 5')

learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=[RootMeanSquaredError()])

print('Check 6')

# Treinamento do modelo

num_epochs = 50
batch_size = 40
# model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, callbacks=[check_point], validation_split=0.1)
history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
print('Check 7')

# Previsões

X_val, Y_val = Data_to_X_Y(val_data, N_lags)

print('Check 8')

Y_pred = model.predict(X_val)

print('Check 9')

# Avaliação do modelo

def calculate_SMAPE(actual, predicted):
    if len(actual) != len(predicted):
            raise ValueError("Arrays devem ter o mesmo tamanho\n")

    abs_diff = np.abs(actual - predicted)
    summ = np.abs(actual) + np.abs(predicted)

    zero_mask = summ == 0
    valid_mask = ~zero_mask
    smape = np.zeros_like(abs_diff, dtype=float)
    smape[valid_mask] = (abs_diff[valid_mask] / summ[valid_mask])
    smape[zero_mask] = 0

    smape = np.mean(smape)

    return round(smape*100, 2)

MSE = MeanSquaredError()
MSE_value = MSE(Y_val, Y_pred).numpy()
print(f'Erro Médio Quadrático de validação (MSE): {MSE_value}')

sMAPE_val = calculate_SMAPE(Y_val,Y_pred) #(np.mean(np.abs(Y_val - Y_pred) / ((abs(Y_pred) + abs( Y_val))/2)) * 100, 2)
print(f'Erro Médio Percentual Absoluto Simétrico de validação (SMAPE): {sMAPE_val}%')

print('Check 10')

train_loss = history.history['loss'][-1]

# Criar baseline aqui:

def avg_baseline(P_table, N):
    Y = []
    for item in P_table.columns:
        y = []
        serie_as_np = np.array(P_table[item])
        for i in range(len(serie_as_np) - N - 28 ):
            row_X = [a for a in serie_as_np[i:i+N]]
            avg = float(np.array(row_X).mean()) 
            row_Y_avg = [avg]*28
            y.append(row_Y_avg)
        Y.append(y)
    Y = np.array(Y)
    Y = Y.transpose(1,2,0)
    return Y

print('Check 11')

Y_pred_baseline = avg_baseline(val_data, N_lags)

print('Check 12')

print('Y_pred_baseline.shape : ', Y_pred_baseline.shape)
print('Y_val.shape : ', Y_val.shape)

print('Check 13')

MSE_baseline = MeanSquaredError()
MSE_value_baseline = MSE_baseline(Y_val, Y_pred_baseline).numpy()
print(f'Erro Médio Quadrático de validação (MSE) com baseline: {MSE_value_baseline}')

sMAPE_val_baseline = calculate_SMAPE(Y_val, Y_pred_baseline) #(np.mean(np.abs(Y_val - Y_pred_baseline) / ((abs(Y_pred_baseline) + abs( Y_val))/2)) * 100, 2)
print(f'Erro Médio Percentual Absoluto Simétrico de validação (SMAPE) com baseline: {sMAPE_val_baseline}%')

print('Check 14')

end = timeit.default_timer()
now = datetime.now()
exec_time = round((end - start)/60,2)

print(f'\nCódigo executado em {now}. Tempo de execução: {exec_time} minutos.\n')

# Coletando métricas:

# with open('../../models/task3/logs/logs_multi_series.txt', 'a') as f:
#     model.summary(print_fn=lambda x: f.write(x + '\n'))
#     f.write(f'Erro Médio Quadrático de validação (MSE) com baseline: {MSE_value_baseline}\n')
#     f.write(f'Erro Médio Percentual Absoluto Simétrico de validação (SMAPE) com baseline: {sMAPE_val_baseline}% \n')
#     f.write(f'MSE de treino (loss function): {train_loss}\n')
#     f.write(f'Leanrnig rate: {learning_rate}\n')
#     f.write(f'Regularização das camadas LSTM: l1\n')
#     f.write(f'Quandidade de dias anteriores usada para treino: {N_lags}\n')
#     f.write(f'Tamanho dos batches: {batch_size}\n')
#     f.write(f'Número de épocas de treinamento: {num_epochs}\n')
#     f.write(f'Erro Médio Percentual Absoluto Simétrico de validação (SMAPE): {sMAPE_val}%\n')
#     f.write(f'Erro Médio Quadrático de validação (MSE): {MSE_value}\n')
#     f.write(f'\nCódigo executado em {now}.\n Tempo de execução: {exec_time} minutos.\n')
#     f.write('-------------------------------------------------------\n\n')
