import poloniex
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

in_out_neurons = 1
hidden_neurons = 300
length_of_sequences = 50

def _load_data(data, n_prev=50):
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

def train_test_split(df, test_size=0.1, n_prev=50):
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)
    return (X_train, y_train), (X_test, y_test)

if __name__ == '__main__':
    polo = poloniex.Poloniex()
    polo.timeout = 2
    rawdata = polo.returnChartData('USDT_BTC',
                                   period=300,
                                   start=time.time() - (60 * 60 * 24) * 180,
                                   end=time.time())

    #価格を調整
    price_data = pd.DataFrame([float(i.get('open')) for i in rawdata])
    mss = MinMaxScaler()
    input_dataframe = pd.DataFrame(mss.fit_transform(price_data))

    (X_train, y_train), (X_test, y_test) = train_test_split(input_dataframe)

    model = Sequential()
    model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="adam",)

    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=0)
    history = model.fit(X_train, y_train, batch_size=600, epochs=10, validation_split=0.1, callbacks=[early_stopping])
    model.save('test.h5')

    pred_data = model.predict(X_test)
    plt.plot(y_test, label='test')
    plt.plot(pred_data, label='pred')
    plt.legend(loc='upper left')
    plt.show()
