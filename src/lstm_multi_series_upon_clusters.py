#!/bin/python3

# <h2 style="color:black; background-color: gray; font-weight: bold">Importando pacotes</h2>

print('Importing modules')

import sys
import os
import timeit
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
start = timeit.default_timer()
import pandas as pd 
import numpy as np 
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras import initializers
from tensorflow.keras.losses import MeanSquaredError
from keras.regularizers import l2, l1
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from lstm_upon_clusters_module import *

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)


#-----------------------------------------------
# ------------------------------- CLI (parte 1)

# just to prevent unbound warnings:
predicted_item_name = 'item'
cluster_number = 42
n_series_training = 1000000000
n_series = 10000000000

if len(sys.argv) < 2:
    print("Uso: python script.py <numero do cluster: (0, 4)> <n_series_training> <n_series_prediction>") # <------------- entrar com número do cluster
    sys.exit(1)

if sys.argv[1] == 'p': #<------ quando 'p' é passado no argumento 1 o programa faz a previsão particular de um item 
        cli_way = 'p'
        if sys.argv[2] != 'f':
            n_series_training = int(sys.argv[2])
        n_series = 1
        predicted_item_name = sys.argv[3] # <------------------- nome do produto que será previsto
else:
    cli_way = 'm'
    cluster_number = int(sys.argv[1])
# ------------------------------- CLI (parte 1)
#-----------------------------------------------

# <h2 style="color:black; background-color: gray; font-weight: bold">Carregando dados preprocessados</h2>

print('Reading data')

pivot_table = pd.read_csv('../data/timeseries.csv', sep=';', low_memory=False)

# pivot_table.iloc[1:1:] = pivot_table.iloc[1:1:].round(2)
# pivot_table = pivot_table.transpose()

#-----------------------------------------------
#-----------------------------------------------
# Clusterização global com K-Means
print('Clustering')

scaler = StandardScaler()
pivot_table_scaled = scaler.fit_transform(pivot_table.iloc[1:,1:])

pivot_table_scaled = pivot_table_scaled.transpose()

inertia = []

# Testar diferentes valores de k (número de clusters)
# for k in range(1, 30):
#     kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
#     kmeans.fit(pivot_table_scaled)
#     inertia.append(kmeans.inertia_)

# # Plotar o gráfico do método Elbow
# plt.plot(range(1, 30), inertia)
# plt.xlabel('Número de Clusters')
# plt.ylabel('Inércia')
# plt.title('Método Elbow por Produto')
# plt.show()

# Gráfico aponta n_clusters=5 ===>
n_clusters = 5
kmeans = KMeans(n_clusters = n_clusters, random_state=0, n_init=10)
cluster_labels = kmeans.fit_predict(pivot_table_scaled)

pivot_table = pivot_table.transpose()

cluster_labels = list(cluster_labels)
pivot_table['cluster'] = cluster_labels

#-----------------------------------------------
#-----------------------------------------------

# fazer aqui a seleção dos itens de acordo com o cluster 
# cluster_number = 4

if cli_way == 'p':
    cluster_number = pivot_table.loc[pivot_table.index == predicted_item_name]['cluster'][0]
    print('Cluster number: ', cluster_number)

cluster = pivot_table[pivot_table['cluster'] == cluster_number]
cluster = cluster.drop('cluster', axis=1)
cluster = cluster.transpose()
cluster_copy = cluster.copy()

# print(cluster.head())

n_series_in_cluster = len(cluster.columns)


# ------------------------------------------------------
# ------------------------------- lendo da CLI:

if sys.argv[1] == 'p':
    if sys.argv[2] == 'f':
        n_series_training = n_series_in_cluster
    else:
        n_series_training = int(sys.argv[2]) # <-------------------- quantidade de séries usadas para treinamento

if sys.argv[1] != 'p':
    if sys.argv[2] == 'f':
        n_series_training = n_series_in_cluster
    else:
        n_series_training = int(sys.argv[2]) # <-------------------- quantidade de séries usadas para treinamento

    if (sys.argv[3] == 'f'):
        n_series = n_series_in_cluster
    else:
        n_series = int(sys.argv[3])          # <-------------------- quantidade de séries previstas

# ------------------------------- fim da leitura da CLI:
# ------------------------------------------------------

sorted_serie = cluster.sum().sort_values(ascending=False)

# print(sorted_serie.head())

# n_series_training = int(n_series_in_cluster/2)

# n_series_training = 200 # <-------------------- quantidade de séries usadas para treinamento
# n_series = 10          # <-------------------- quantidade de séries previstas

top_sales = sorted_serie.head(n_series)
top_sales_items = top_sales.index
top_series = cluster[top_sales_items]
# print(top_series.shape)

scaler = MinMaxScaler()
# pivot_table_renormalized = pd.DataFrame(scaler.fit_transform(top_series), columns=top_sales_items)
pivot_table_renormalized = pd.DataFrame(scaler.fit_transform(cluster), columns=cluster.columns)
# pivot_table_renormalized = pd.DataFrame(cluster, columns=cluster.columns) #not renormalized! But we kept the same name!!!

# <h2 style="color:black; background-color: gray; font-weight: bold">Preparando dados para treino</h2>

# Divisão em conjuntos de treinamento, validação e teste

print('Splitting data into training and test')

pivot_table_renormalized.reset_index(drop=True, inplace=True)
train_size = int(len(pivot_table_renormalized) * 0.8)
train_plus_val_size = int(len(pivot_table_renormalized) * 0.9)
train_data = pivot_table_renormalized[:train_size]
# val_data = pivot_table_renormalized[train_size:]
val_data = pivot_table_renormalized[train_size:train_plus_val_size]
test_data = pivot_table_renormalized[train_plus_val_size:]

print('===================================')
print('tipos de dados: treino, val e teste')
for data in [train_data, val_data, test_data]:
    print(data.shape)
print('===================================')

# Preparação dos dados de treinamento, validação e teste
# Vamos usar os dados dos últimos N dias para prever os próximos 28

# print('colunas:', train_data.columns)


N_lags = 20 
horizon = 4

#Usando todas as séries para prever todas as series:
# X_train, Y_train = Data_to_X_Y(train_data, N_lags, horizon)

#Usando todas as series para prever apenas algumas:
if cli_way == 'm':
    X_train, Y_train = Data_to_X(train_data, N_lags, horizon, n_series_training), Data_to_Y(train_data, N_lags, horizon, n_series)
# if cli_way == 'p':
else:
    X_train, Y_train = Data_to_X_cli_way_p(train_data, N_lags, horizon, n_series_training, predicted_item_name), Data_to_Y_cli_way_p(train_data, N_lags, horizon, predicted_item_name)


print('X_train:', X_train.shape)
print('Y_train:', Y_train.shape)

# print('fim da linha')
# sys.exit(1)

n_instances = X_train.shape[0]

print('Number of training  instances:', n_instances)

# <h2 style="color:black; background-color: gray; font-weight: bold">Modelo:</h2>

# ***************************************** Arquitetura do Modelo:

print('Setting Model Architecture')

_lambda = 0.01
initializer = initializers.RandomNormal(mean=0.0, stddev=0.0, seed=seed)
lstm_number_of_neurons = 100
lstm_layer_to_dense = LSTM(lstm_number_of_neurons, kernel_regularizer=l1(_lambda), kernel_initializer= initializer)
lstm_layer_to_lstm = LSTM(lstm_number_of_neurons, kernel_regularizer=l1(_lambda), kernel_initializer= initializer, return_sequences=True)

model = Sequential()

model.add(InputLayer((N_lags, n_series_training)))

# model.add(lstm_layer_to_lstm)

# model.add(LSTM(80, kernel_regularizer=l1(_lambda), return_sequences=True))
model.add(Dropout(0.2))
model.add(lstm_layer_to_dense)
# model.add(Dense(50, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(200, activation='relu'))

last_layer_size = int(horizon*n_series) #int(28*n_series/35)

# model.add(Dropout(0.2))

model.add(Dense(last_layer_size, activation='linear')) 
model.add(Reshape((horizon, n_series)))

model.summary()

# ******************************************************************

# Crie um diretório para salvar os modelos
os.makedirs('./models/', exist_ok=True)

# Callback personalizado
custom_checkpoint = SaveBestModel(cluster_number, n_series_training, n_series, predicted_item_name)

# check_point = ModelCheckpoint('./models/', filename=generate_model_name, save_best_only=True, monitor='val_loss')

print('Compiling')

learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=[RootMeanSquaredError()])

print('Training')
print('n_series_training:', n_series_training)
print('predicted_item_name:', predicted_item_name)
# Treinamento do modelo

num_epochs = 30
batch_size = 20
history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, callbacks=[custom_checkpoint], validation_split=0.1, verbose=0)

# model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, callbacks=[check_point], validation_split=0.1)
# history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
print('Constructing Validation Data')

# Previsões
# print('train_data')
# print(train_data.head())
# print('val_data')
# print(val_data.head())

# val_data.reset_index(inplace=True)
# X_val, Y_val = Data_to_X_Y(val_data, N_lags, horizon)
#Usando todas as series para prever apenas algumas:
# X_val, Y_val = Data_to_X(val_data, N_lags, horizon), Data_to_Y(val_data, N_lags, horizon, n_series)
if cli_way == 'm':
    X_val, Y_val = Data_to_X(pivot_table_renormalized, N_lags, horizon, n_series_training), Data_to_Y(pivot_table_renormalized, N_lags, horizon, n_series)
else:
    X_val, Y_val = Data_to_X_cli_way_p(pivot_table_renormalized, N_lags, horizon, n_series_training, predicted_item_name), Data_to_Y_cli_way_p(pivot_table_renormalized, N_lags, horizon, predicted_item_name)


if cli_way == 'm':
    X_test, Y_test = Data_to_X(test_data, N_lags, horizon, n_series_training), Data_to_Y(test_data, N_lags, horizon, n_series)
else:
    X_test, Y_test = Data_to_X_cli_way_p(test_data, N_lags, horizon, n_series_training, predicted_item_name), Data_to_Y_cli_way_p(test_data, N_lags, horizon, predicted_item_name)

print('X_val:', X_val.shape)
print('Y_val:', Y_val.shape)


print('Predicting')

Y_pred = model.predict(X_val)
# print('Prediction shape:', Y_pred.shape)

print('Calculating metrics')

# Avaliação do modelo


MSE = MeanSquaredError()
MSE_value = MSE(Y_val, Y_pred).numpy()

print('====================================================')
print(f'Cluster number: {cluster_number}')
print('Número de épocas de treinamento:', num_epochs)
print(f'MSE de validação: {MSE_value}')

sMAPE_val = calculate_SMAPE(Y_val,Y_pred) #(np.mean(np.abs(Y_val - Y_pred) / ((abs(Y_pred) + abs( Y_val))/2)) * 100, 2)
print(f'SMAPE de validação: {sMAPE_val}%')


train_loss = history.history['loss'][-1]

# Y_pred_baseline = avg_baseline(val_data, N_lags, n_series)
if cli_way == 'm':
    Y_pred_baseline = avg_baseline(pivot_table_renormalized, N_lags, n_series)
else: 
    Y_pred_baseline = avg_baseline_cli_way_p(pivot_table_renormalized, N_lags)

# print('Y_pred_baseline.shape : ', Y_pred_baseline.shape)
# print('Y_val.shape : ', Y_val.shape)

MSE_baseline = MeanSquaredError()
MSE_value_baseline = MSE_baseline(Y_val, Y_pred_baseline).numpy()
print(f'MSE de validação do baseline: {MSE_value_baseline}')

sMAPE_val_baseline = calculate_SMAPE(Y_val, Y_pred_baseline) #(np.mean(np.abs(Y_val - Y_pred_baseline) / ((abs(Y_pred_baseline) + abs( Y_val))/2)) * 100, 2)
print(f'SMAPE de validação do baseline: {sMAPE_val_baseline}%')

print('====================================================')

#--------------------------------------------
#plotar gráficos aqui

print('Plotting top 1 item')


if cli_way == 'm':
    item_name_inside_list, shown_serie = get_top_items(pivot_table_renormalized, 2)
    item_name = item_name_inside_list[0]
else:
    item_name = predicted_item_name
    shown_serie = pivot_table_renormalized[predicted_item_name]

print('top_1_item:', item_name)


# -----------------------------------------------------------------------------

# being wild ...
_ , top_series = get_top_items(pivot_table_renormalized, n_series)

if cli_way == 'm':
    col_position = top_series.columns.get_loc(item_name)
else:
    col_position = 0

print('col_position:', col_position)

prediction_top1_item_first_4 = [Y_pred[0,0,col_position], 
        (Y_pred[0,1,col_position] + Y_pred[1,0,col_position])/2 , 
        (Y_pred[0,2,col_position] + Y_pred[1,1,col_position] + Y_pred[2,0,col_position])/3,
        (Y_pred[0,3,col_position] + Y_pred[1,2,col_position] + Y_pred[2,1,col_position] + Y_pred[3,0,col_position])/4]

prediction_top1_item_rest = []
pred_length = len(Y_pred)
for i in range(0,pred_length-horizon):
    P = (Y_pred[i,3,col_position] + Y_pred[i+1,2,col_position] + Y_pred[i+2,1,col_position] + Y_pred[i+3,0,col_position])/4
    prediction_top1_item_rest.append(P)

last_position = pred_length - 1

prediction_top1_item_last_terms = [
        (Y_pred[last_position-3, 3,col_position] + Y_pred[last_position-2, 2,col_position] + Y_pred[last_position-1, 1,col_position] + Y_pred[last_position,0,col_position])/4 ,
        (Y_pred[last_position-2, 3,col_position] + Y_pred[last_position-1, 2,col_position] + Y_pred[last_position-0, 1,col_position])/3,
        (Y_pred[last_position, 2,col_position] + Y_pred[last_position-1, 3,col_position])/2 , 
        Y_pred[last_position, 3,col_position] ]

full_rediction_1_item = prediction_top1_item_first_4 + prediction_top1_item_rest + prediction_top1_item_last_terms

# -----------------------------------------------------------------------------

#serie do baseline:

prediction_top1_item_first_4_baseline = [Y_pred_baseline[0,0,col_position], 
        (Y_pred_baseline[0,1,col_position] + Y_pred_baseline[1,0,col_position])/2 , 
        (Y_pred_baseline[0,2,col_position] + Y_pred_baseline[1,1,col_position] + Y_pred_baseline[2,0,col_position])/3,
        (Y_pred_baseline[0,3,col_position] + Y_pred_baseline[1,2,col_position] + Y_pred_baseline[2,1,col_position] + Y_pred_baseline[3,0,col_position])/4]

prediction_top1_item_rest_baseline = []
pred_length = len(Y_pred)
for i in range(0,pred_length-horizon):
    P = (Y_pred_baseline[i,3,col_position] + Y_pred_baseline[i+1,2,col_position] + Y_pred_baseline[i+2,1,col_position] + Y_pred_baseline[i+3,0,col_position])/4
    prediction_top1_item_rest_baseline.append(P)

last_position_baseline = pred_length - 1

prediction_top1_item_last_terms_baseline = [
        (Y_pred_baseline[last_position-3, 3,col_position] + Y_pred_baseline[last_position-2, 2,col_position] + Y_pred_baseline[last_position-1, 1,col_position] + Y_pred_baseline[last_position,0,col_position])/4 ,
        (Y_pred_baseline[last_position-2, 3,col_position] + Y_pred_baseline[last_position-1, 2,col_position] + Y_pred_baseline[last_position-0, 1,col_position])/3,
        (Y_pred_baseline[last_position, 2,col_position] + Y_pred_baseline[last_position-1, 3,col_position])/2 , 
        Y_pred_baseline[last_position, 3,col_position] ]

full_rediction_1_item_baseline = prediction_top1_item_first_4_baseline + prediction_top1_item_rest_baseline + prediction_top1_item_last_terms_baseline

# -----------------------------------------------------------------------------

scaled_prediction_top1_item = np.array(full_rediction_1_item)
scaled_prediction_top1_item_baseline = np.array(full_rediction_1_item_baseline)

# Undoing Min_Max_Scaler for just 1 column:

min_col = cluster_copy.min().min()
max_col = cluster_copy.max().max()
un_scaled_prediction_top1_item = (max_col - min_col)* scaled_prediction_top1_item + min_col
un_scaled_prediction_top1_item_baseline = (max_col - min_col)* scaled_prediction_top1_item_baseline + min_col

if cli_way == 'm':
    un_scaled_top1_serie_train = (max_col - min_col) * shown_serie[:train_plus_val_size] + min_col
    un_scaled_top1_serie_test = (max_col - min_col) * shown_serie[train_size:] + min_col
else: 
    un_scaled_train_serie = (max_col - min_col) * shown_serie[:train_plus_val_size] + min_col
    un_scaled_test_serie = (max_col - min_col) * shown_serie[train_plus_val_size:] + min_col

# un_scaled_prediction_top1_item = pd.Series(un_scaled_prediction_top1_item)
# un_scaled_prediction_top1_item.index = range(train_size, train_size + len(un_scaled_prediction_top1_item))

# print(un_scaled_prediction_top1_item)

end = timeit.default_timer() #<--------------- terminando contagem do tempo antes da Visualização dos gráficos
now = datetime.now()
exec_time = round((end - start)/60,2)

if cli_way == 'm':
    plt.plot(un_scaled_top1_serie_train, label='Dados de Treinamento')
    plt.plot(un_scaled_top1_serie_test, label='Dados de Teste')
    plt.title(f'{item_name}: mais vendido do cluster {cluster_number} - Dados semanais')
else: 
    plt.title(f'Vendas globais de {item_name}: cluster {cluster_number} - Dados semanais')
    plt.plot(un_scaled_train_serie, label='Dados de Treinamento')
    plt.plot(un_scaled_test_serie, label='Dados de Teste')
plt.plot(un_scaled_prediction_top1_item, label='Previsão do modelo LSTM')
plt.plot(un_scaled_prediction_top1_item_baseline, label='Baseline (média simples)')
plt.legend()
plt.xlabel('Semanas')
plt.ylabel('Unidades vendidas') 

plt.show()
# plt.savefig(f'prediction_top1_cluster{cluster_number}_{n_series_training}_to_{n_series}.png', dpi=300)

#--------------------------------------------

print(f'\nCódigo executado em {now}. Tempo de execução: {exec_time} minutos.\n')

# Coletando métricas:

# with open('./logs_lstm_upon_clusters_2.log', 'a') as f:
#     f.write(f'Usando todos os items. E prevendo apenas as top 100.\n')
#     f.write(f'Cluster: {cluster_number}\n')
#     f.write(f'MSE de treino (loss function): {train_loss}\n')
#     f.write(f'MSE de validação: {MSE_value}\n')
#     f.write(f'MSE de validação do baseline: {MSE_value_baseline}\n')
#     f.write(f'SMAPE de validação: {sMAPE_val}%\n')
#     f.write(f'SMAPE de validação do baseline: {sMAPE_val_baseline}% \n')
#     f.write(f'\nCódigo executado em {now}.\n Tempo de execução: {exec_time} minutos.\n')
#     f.write('-------------------------------------------------------\n\n')
