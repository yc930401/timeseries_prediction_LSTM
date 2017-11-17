import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

def get_data(train_size=365*24, time_steps = 1):

    dataset = pd.read_csv('Beijing_PM25_processed.csv', header=0, index_col=0)
    dataset = pd.get_dummies(dataset)
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    x_data = []
    y_data = []
    if time_steps == 1:
        for i in range(len(scaled)-1):
            x_data.append([scaled[i]])
            y_data.append([scaled[i+1, 0]])
    else:
        for i in range(len(scaled) - time_steps - 1):
            x_data.append(scaled[i: i+time_steps])
            y_data.append(scaled[i+time_steps+1, 0])

    x_train, y_train = np.array(x_data[:train_size]), np.array(y_data[:train_size])
    x_test, y_test = np.array(x_data[train_size:]), np.array(y_data[train_size:])

    return scaler, dataset.index.get_values(), x_train.reshape(x_train.shape[0], time_steps, x_train.shape[-1]), \
           y_train.reshape(y_train.shape[0], 1), \
           x_test.reshape(x_test.shape[0], time_steps, x_test.shape[-1]), \
           y_test.reshape(y_test.shape[0], 1)
