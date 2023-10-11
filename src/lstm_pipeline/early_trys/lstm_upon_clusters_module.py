from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd

# Classe de callback personalizada para salvar o melhor modelo com base na val_loss
class SaveBestModel(Callback):
    def __init__(self, cluster_number, n_series_training, n_series, product_name):
        super().__init__()
        self.cluster_number = cluster_number
        self.n_series_training = n_series_training
        self.n_series = n_series
        self.best_val_loss = float('inf')  # Inicialize com um valor infinito
        self.best_model = None
        self.product_name = product_name

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model = self.model  # Atualize o melhor modelo
            model_name = f'model_{self.product_name}_cluster_{self.cluster_number}_{self.n_series_training}_to_{self.n_series}_val_loss_{val_loss:.4f}.keras'
            self.model.save(os.path.join('../models/', model_name), save_format='keras')


def get_top_items(P_table, top_n):
    if top_n > 2:
        Sorted_serie = pd.DataFrame(P_table).sum().sort_values(ascending=False)
        Top_sales = Sorted_serie.head(top_n)
        Top_sales_items = list(Top_sales.index)
        Top_series = pd.DataFrame(P_table)[Top_sales_items]
    else: 
        Top_sales_items = []
        Top_series = pd.DataFrame()
    return Top_sales_items, Top_series

def Data_to_X_Y(P_table, N, horizon):
    X = []
    Y = []
    for item in P_table.columns:
        x = []
        y = []
        serie_as_np = np.array(P_table[item])
        for i in range(0, len(serie_as_np) - N - horizon ):
            row_X = [a for a in serie_as_np[i:i+N]]
            row_Y = [a for a in serie_as_np[i+N:i+N+horizon]]
            x.append(row_X)
            y.append(row_Y)
        X.append(x)
        Y.append(y)
    X, Y = np.array(X) , np.array(Y)
    X = X.transpose(1,2,0)
    Y = Y.transpose(1,2,0)
    return X, Y

def Data_to_X(P_table, N, horizon, n_series_training):
    X = []
    top_sales_items, top_series = get_top_items(P_table, n_series_training)
    for item in top_sales_items:
        x = []
        serie_as_np = np.array(top_series[item])
        for i in range(0, len(serie_as_np) - N - horizon ):
            row_X = [a for a in serie_as_np[i:i+N]]
            x.append(row_X)
        X.append(x)
    X = np.array(X)
    X = X.transpose(1,2,0)
    return X

def Data_to_X_cli_way_p(P_table, N, horizon, n_series_training, predicted_item_name):
    X = []
    actual_number_of_other_series = n_series_training - 1
    top_sales_items, top_series = get_top_items(P_table, actual_number_of_other_series)
    main_serie = P_table[predicted_item_name]
    main_serie_as_np = np.array(main_serie)
    x0 = []
    for i in range(0, len(main_serie_as_np) - N - horizon ):
        row_X = [a for a in main_serie_as_np[i:i+N]]
        x0.append(row_X)
    X.append(x0)
    for item in top_sales_items:
        x = []
        serie_as_np = np.array(top_series[item])
        for i in range(0, len(serie_as_np) - N - horizon ):
            row_X = [a for a in serie_as_np[i:i+N]]
            x.append(row_X)
        X.append(x)
    print(len(X))
    X = np.array(X)
    X = X.transpose(1,2,0)
    return X

def Data_to_Y(P_table, N, horizon, n_series_prediction):
    Y = []
    top_sales_items, top_series = get_top_items(P_table, n_series_prediction)
    top_series = pd.DataFrame(P_table)[top_sales_items]
    for item in top_sales_items:
        y = []
        serie_as_np = np.array(top_series[item])
        for i in range(0, len(serie_as_np) - N - horizon ):
            row_Y = [a for a in serie_as_np[i+N:i+N+horizon]]
            y.append(row_Y)
        Y.append(y)
    Y = np.array(Y)
    Y = Y.transpose(1,2,0)
    return Y

def Data_to_Y_cli_way_p(P_table, N, horizon, predicted_item_name):
    Y = []
    serie = P_table[predicted_item_name]
    serie_as_np = np.array(serie)
    y = []
    for i in range(0, len(serie_as_np) - N - horizon ):
        row_Y = [a for a in serie_as_np[i+N:i+N+horizon]]
        y.append(row_Y)
    Y.append(y)
    Y = np.array(Y)
    Y = Y.transpose(1,2,0)
    return Y

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


def avg_baseline(P_table, N, n_series_prediction):
    Y = []
    top_sales_items, top_series = get_top_items(P_table, n_series_prediction)
    top_series = P_table[top_sales_items]
    for item in top_sales_items:
        y = []
        serie_as_np = np.array(top_series[item])
        for i in range(0, len(serie_as_np) - N - horizon ):
            row_X = [a for a in serie_as_np[i:i+N]]
            avg = float(np.array(row_X).mean()) 
            row_Y_avg = [avg]*horizon
            y.append(row_Y_avg)
        Y.append(y)
    Y = np.array(Y)
    Y = Y.transpose(1,2,0)
    return Y


def avg_baseline_cli_way_p(P_table, N):
    Y = []
    serie = P_table[predicted_item_name]
    serie_as_np = np.array(serie)
    y = []
    for i in range(0, len(serie_as_np) - N - horizon ):
        row_X = [a for a in serie_as_np[i:i+N]]
        avg = float(np.array(row_X).mean()) 
        row_Y_avg = [avg]*horizon
        y.append(row_Y_avg)
    Y.append(y)
    Y = np.array(Y)
    Y = Y.transpose(1,2,0)
    return Y

