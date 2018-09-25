import poloniex
import csv
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

in_neurons = 2#入力数
out_neurons = 1#出力数
hidden_neurons = 300
length_of_sequences = 36#波形の１セット（３６＝１８０分）
input_file_name = "trance20180922.csv"#学習データ

def _load_data(data, n_prev=36):
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())

    print(docY)
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY

def train_test_split(narray, test_size=0.1, n_prev=36):
    ntrn = round(len(narray) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(narray.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(narray.iloc[ntrn:], n_prev)
    return (X_train, y_train), (X_test, y_test)

if __name__ == '__main__':
    #データの読み込み
    csv_data = open(input_file_name, "r", encoding="ms932", errors="", newline="" )
    csv_file = csv.reader(csv_data, delimiter=",", doublequote=True, lineterminator="", quotechar='"', skipinitialspace=True)

    #価格を正規化
    price_data = [i[1:] for i in csv_file]
    price_data = pd.DataFrame(price_data)
    mss = MinMaxScaler(feature_range=(0,10))
    input_dataframe = pd.DataFrame(mss.fit_transform(price_data))
    (X_train, y_train), (X_test, y_test) = train_test_split(input_dataframe)

    #モデル構成
    model = Sequential()
    model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_neurons), return_sequences=False))
    model.add(Dense(out_neurons))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="adam",)

    #学習の実施
    epochs = 10
    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=0)
    history = model.fit(X_train, y_train, batch_size=500, epochs=epochs, validation_split=0.1, callbacks=[early_stopping])

    #学習結果の確認
    pred_data = model.predict(X_test)
    plt.plot(y_test, label='test')
    plt.plot(pred_data, label='pred')
    plt.legend(loc='upper left')
    plt.show()
