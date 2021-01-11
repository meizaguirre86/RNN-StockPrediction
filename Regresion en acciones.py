import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import datetime
from tensorflow.python.keras.callbacks import TensorBoard

plt.style.use('ggplot')

#VARIABLES
TIME_STEPS = 30
STOCK_TICKER = 'GOOGL'
STOCK_PERIOD = 'max'
BATCH_SIZE = 20
FEATURES = ['Close', 'Volume']
LEARNING_RATE = 0.0001
EPOCHS = 2
COL_Y = 0
LOG_DIR = r"C:\Users\meiza\Python\logs\logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


#FUNCTION CREATE 'X' AND 'y'
def build_timeseries(mat, y_col_index):
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))
    y = np.zeros((dim_0,))

    for i in range(dim_0):
        x[i] = mat[i:TIME_STEPS + i]
        y[i] = mat[TIME_STEPS + i, y_col_index]
    print("length of time-series i/o", x.shape, y.shape)
    return x, y

#FUNCIÓN PARA ELIMINAR SAMPLES SOBRANTES EN CASO DE QUE EL DATASET NO SEA DIVISIBLE POR EL BATCH_SIZE
def trim_dataset(mat):
    no_of_rows_drop = mat.shape[0]%BATCH_SIZE
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

#GET DATAFRAME, CONVERT DATE INDEX TO SERIES AND REMOVE IT
ticker=yf.Ticker(STOCK_TICKER)
data = ticker.history(period=STOCK_PERIOD)
data['Date'] = data.index.to_series()
data.reset_index(drop = True, inplace = True)

#SPLIT DATAFRAME INTO TRAIN AND TEST FRAMES AND SCALE FEATURES
data_train, data_test = train_test_split(data, train_size = 0.8, test_size = 0.2, shuffle=False)
min_max_scaler = MinMaxScaler()
X = data.loc[:,FEATURES].values
min_max_scaler.fit(X)
X_train = min_max_scaler.transform(data_train.loc[:,FEATURES].values)
X_test = min_max_scaler.transform(data_test.loc[:,FEATURES].values)

#TRANSFORMAMOS LOS DATOS PARA QUE SE PUEDA ENTRENAR EL RNN
X_t, y_t = build_timeseries(X_train,COL_Y)
X_t = trim_dataset(X_t)
y_t = trim_dataset(y_t)

#CREAMOS MODELO Y COMPILAMOS
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, X_t.shape[2]), dropout=0.0, recurrent_dropout=0.0, stateful=True, kernel_initializer='random_uniform'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(20,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
optimizer = tf.keras.optimizers.RMSprop(lr=LEARNING_RATE)
model.compile(loss='mean_squared_error', optimizer=optimizer)

#CREAMOS MODELOS Y CAPAS
model = tf.keras.models.Sequential()
# (batch_size, timesteps, data_dim)
model.add(tf.keras.layers.LSTM(100, batch_input_shape=(BATCH_SIZE, TIME_STEPS, X_t.shape[2]),
                    dropout=0.0, recurrent_dropout=0.0, stateful=True, return_sequences=True,
                    kernel_initializer='random_uniform'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.LSTM(60, dropout=0.0))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(20,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
optimizer = tf.keras.optimizers.RMSprop(lr=LEARNING_RATE)
# optimizer = optimizers.SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=optimizer)

#CREAMOS DATAFRAMES
X_temp, y_temp = build_timeseries(X_test, COL_Y)
X_val, X_test_t = np.split(trim_dataset(X_temp), 2)
y_val, y_test_t = np.split(trim_dataset(y_temp), 2)

tensorboard = TensorBoard(log_dir=LOG_DIR)

history = model.fit(X_t, y_t, epochs=EPOCHS, verbose=2, batch_size=BATCH_SIZE,
                    shuffle=False, validation_data=(trim_dataset(X_val), trim_dataset(y_val)), callbacks=[tensorboard])



