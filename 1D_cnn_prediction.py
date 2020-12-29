import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
import keras.metrics
from sklearn.model_selection import train_test_split

# loading data into df's and series
precipitation_df = pd.read_csv("6077P.csv",
                               dtype={'Precipitation': np.float64})
datetimep = precipitation_df["Datetime"]
precipitation = precipitation_df["Precipitation"]
discharge_df = pd.read_csv("5096D.csv",
                           dtype={'Discharge': np.float64},
                           index_col=0)
datetimed = discharge_df["Datetime"]
discharge = discharge_df["Discharge"]

p1 = datetimep[datetimep == "01/07/2017 07:00:00"].index[0]
p2 = datetimep[datetimep == "01/09/2017 15:15:00"].index[0]
p3 = datetimep[datetimep == "01/16/2019 10:15:00"].index[0]
p4 = datetimep[datetimep == "01/18/2019 13:00:00"].index[0]
event_rainfall = list(precipitation_df["Precipitation"].iloc[p1:p2]) +\
                 list(precipitation_df["Precipitation"].iloc[p3:p4])
d1 = datetimed[datetimed == "01/07/2017 10:00:00"].index[0]
d2 = datetimed[datetimed == "01/09/2017 15:15:00"].index[0]
d3 = datetimed[datetimed == "01/16/2019 13:15:00"].index[0]
d4 = datetimed[datetimed == "01/18/2019 13:00:00"].index[0]
event = list(discharge_df["Discharge"].iloc[d1:d2]) +\
        list(discharge_df["Discharge"].iloc[d3:d4])
X = []
Y = []
# turning data into [a, b, c, d], [e] where a, b, c, d are precipitation, e is
# discharge
for i in range(len(event)):
    x = [event_rainfall[i], event_rainfall[i + 1], event_rainfall[i + 2],
         event_rainfall[i + 3], event_rainfall[i + 4], event_rainfall[i + 5],
         event_rainfall[i + 6], event_rainfall[i + 7], event_rainfall[i + 8],
         event_rainfall[i + 9], event_rainfall[i + 10],
         event_rainfall[i + 11]]
    y = event[i]
    X.append(x)
    Y.append(y)
X = np.array(X)
Y = np.array(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# normalize data


# beginning of model

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_steps = 12
n_features = 1
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
# define model
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=3, activation='relu',
                 input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError(),
                       keras.metrics.MeanAbsolutePercentageError()])
# fit model
model.fit(X_train, Y_train, epochs=1000, verbose=0)
# demonstrate prediction
predictions = []
for index in range(len(X)):
    x_input = X[index]
    x_input = x_input.reshape((1, 12, 1))
    yhat = model.predict(x_input)
    predictions.append(yhat)
    print(yhat)
predictions = np.array(predictions)
print(model.evaluate(X_test.reshape(X_test.shape[0], X_test.shape[1],
                                    n_features), Y_test))
plt.plot(range(len(Y)), Y)
plt.plot(range(0, len(predictions)), predictions.reshape((len(event), 1)))
plt.show()
