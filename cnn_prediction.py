import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import keras.metrics
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

# loading data into df's and series
values = {}
gauges = np.array([])


def load_csv(station_no, file_name, data_type):
    values.update({station_no: pd.read_csv(file_name,
                                           parse_dates=True,
                                           dtype={data_type: np.float64})})


load_csv(6048, "6048P.csv", "Precipitation")
load_csv(6053, "6053P.csv", "Precipitation")
load_csv(6077, "6077P.csv", "Precipitation")
load_csv(6079, "6079P.csv", "Precipitation")
load_csv(6100, "6100P.csv", "Precipitation")
load_csv(6108, "6108P.csv", "Precipitation")
load_csv(6121, "6121P.csv", "Precipitation")
load_csv(6125, "6125P.csv", "Precipitation")
load_csv(5024, "5024D.csv", "Discharge")
load_csv(5025, "5025D.csv", "Discharge")
load_csv(5027, "5027D.csv", "Discharge")
load_csv(5029, "5029D.csv", "Discharge")
load_csv(5030, "5030D.csv", "Discharge")
load_csv(5096, "5096D.csv", "Discharge")


def load_event(station, start_time, end_time, data_type, arr):
    t1 = (values[station]["Datetime"])[(values[station]["Datetime"])
                                       == start_time].index[0]
    t2 = (values[station]["Datetime"])[(values[station]["Datetime"])
                                       == end_time].index[0]
    gauge = np.array(values[station][data_type].iloc[t1:t2])
    arr = np.concatenate((arr, gauge))
    return arr


# load event for all 14 gauges
start = "12/30/2005 10:15:00"
end = "01/01/2006 11:45:00"
gauges = load_event(6048, start, end, "Precipitation", gauges)
gauges = load_event(6053, start, end, "Precipitation", gauges)
gauges = load_event(6077, start, end, "Precipitation", gauges)
gauges = load_event(6079, start, end, "Precipitation", gauges)
gauges = load_event(6100, start, end, "Precipitation", gauges)
gauges = load_event(6108, start, end, "Precipitation", gauges)
gauges = load_event(6121, start, end, "Precipitation", gauges)
gauges = load_event(6125, start, end, "Precipitation", gauges)
gauges = load_event(5024, start, end, "Discharge", gauges)
gauges = load_event(5025, start, end, "Discharge", gauges)
gauges = load_event(5027, start, end, "Discharge", gauges)
gauges = load_event(5029, start, end, "Discharge", gauges)
gauges = load_event(5030, start, end, "Discharge", gauges)
gauges = load_event(5096, start, end, "Discharge", gauges)
gauges = gauges.reshape((14, 198))
# normalize data (gray-scale image)
X = normalize(gauges)

# beginning of model
model = Sequential()
model.add(Conv2D(filters=5, kernel_size=(3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(16, activation="relu"))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError(),
                       keras.metrics.MeanAbsolutePercentageError()])


# turning data into [a, b, c, d], [e] where a, b, c, d are precipitation, e is
# discharge
# for i in range(len(event)):
#     x = [event_rainfall[i], event_rainfall[i + 1], event_rainfall[i + 2],
#          event_rainfall[i + 3], event_rainfall[i + 4], event_rainfall[i + 5],
#          event_rainfall[i + 6], event_rainfall[i + 7], event_rainfall[i + 8],
#          event_rainfall[i + 9], event_rainfall[i + 10],
#          event_rainfall[i + 11]]
#     y = event[i]
#     X.append(x)
#     Y.append(y)
# X = np.array(X)
# Y = np.array(Y)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# basic dense layer model
# model = Sequential()
# model.add(Dense(64, activation="relu"))
# model.add(Dense(4, activation="relu"))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse',
#               metrics=[keras.metrics.RootMeanSquaredError(),
#                        keras.metrics.MeanAbsolutePercentageError()])
# model.fit(X_train, Y_train, epochs=2000, verbose=0)
# predictions = []
# for index in range(len(X)):
#     x_input = X[index]
#     x_input = x_input.reshape((1, 12))
#     yhat = model.predict(x_input)
#     predictions.append(yhat)
#     print(yhat)
# predictions = np.array(predictions)
# print(model.evaluate(X, Y))
# plt.plot(range(len(Y)), Y)
# plt.plot(range(0, len(predictions)), predictions.reshape((400, 1)))
# plt.show()

# 1D CNN model
# reshape from [samples, timesteps] into [samples, timesteps, features]
# n_steps = 12
# n_features = 1
# X_train = X_train.reshape([X.shape[0], X.shape[1], n_features])
# model = Sequential()
# model.add(Conv1D(filters=16, kernel_size=3, activation='relu',
#                  input_shape=(n_steps, n_features)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse',
#               metrics=[keras.metrics.RootMeanSquaredError(),
#                        keras.metrics.MeanAbsolutePercentageError()])
# # fit model
# model.fit(X_train, Y_train, epochs=1000, verbose=0)
# # demonstrate prediction
# predictions = []
# for index in range(len(X)):
#     x_input = X[index]
#     x_input = x_input.reshape((1, 12, 1))
#     yhat = model.predict(x_input)
#     predictions.append(yhat)
#     print(yhat)
# predictions = np.array(predictions)
# print(model.evaluate(X_test.reshape(X_test.shape[0], X_test.shape[1],
#                                     n_features), Y_test))
# plt.plot(range(len(Y)), Y)
# plt.plot(range(0, len(predictions)), predictions.reshape((400, 1)))
# plt.show()
