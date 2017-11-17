from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import numpy as np
from P2_get_data import get_data

epochs = 150
batch_size = 100
time_steps = 2


def LSTM_train(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                        verbose=1, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig('loss_{}.png'.format(time_steps))
    plt.show()

    return model


def LSTM_predict(model, scaler, index, x_train, x_test):
    x_data = np.concatenate((x_train, x_test), axis = 0)
    y_pred = model.predict(x_data)
    x_data = x_data[:, 0, :].reshape(x_data.shape[0], x_data.shape[2])

    all_true = x_data
    all_pred = np.concatenate((y_pred, x_data[:,1:]),axis=1)

    inv_y_pred = scaler.inverse_transform(all_true)[:, 0]
    inv_y_true = scaler.inverse_transform(all_pred)[:, 0]
    x = np.arange(0, x_data.shape[0])
    start = x_train.shape[0]

    plt.plot(x[start: start+100], inv_y_pred[start: start+100], label='pred')
    plt.plot(x[start: start + 100], inv_y_true[start: start + 100], label='true')
    plt.ylabel('Pollution')
    plt.xlabel('Datetime start from {}'.format(index[start]))
    plt.legend()
    plt.savefig('predict_{}.png'.format(time_steps))
    plt.show()

    RMSE(x_train.shape[0], inv_y_pred, inv_y_true)


def RMSE(size, inv_y_pred, inv_y_true):
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y_true[size:], inv_y_pred[size:]))
    print('Test RMSE: %.3f' % rmse)


if __name__ == '__main__':
    scaler, index, x_train, y_train, x_test, y_test = get_data(train_size = 365*24*3, time_steps=time_steps)
    print('Training data size: ', x_train.shape, y_train.shape)
    print('Validation data size: ', x_test.shape, y_test.shape)
    model = LSTM_train(x_train, y_train, x_test, y_test)
    LSTM_predict(model, scaler, index, x_train, x_test)