import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D
from keras import regularizers
import keras.metrics
import time
from scipy.signal import find_peaks

saratoga = {"p_csv": "6077P.csv",
            "d_csv": "5096D.csv",
            "02/11/1992": [np.datetime64("1992-02-11T16:15:00"),
                           np.datetime64("1992-02-14T02:00:00")],
            "02/18/1996": [np.datetime64("1996-02-18T19:15:00"),
                           np.datetime64("1996-02-23T15:30:00")],
            "12/09/1996": [np.datetime64("1996-12-09T14:30:00"),
                           np.datetime64("1996-12-13T08:45:00")],
            "12/31/1996": [np.datetime64("1996-12-31T22:00:00"),
                           np.datetime64("1997-01-04T12:30:00")],
            "02/03/1998": [np.datetime64("1998-02-02T10:45:00"),
                           np.datetime64("1998-02-03T20:15:00")],
            "02/07/1998": [np.datetime64("1998-02-07T08:30:00"),
                           np.datetime64("1998-02-09T05:00:00")],
            "12/16/2002": [np.datetime64("2002-12-16T01:00:00"),
                           np.datetime64("2002-12-19T02:00:00")],
            "02/25/2004": [np.datetime64("2004-02-25T00:15:00"),
                           np.datetime64("2004-03-03T08:45:00")],
            "12/30/2005": [np.datetime64("2005-12-30T18:00:00"),
                           np.datetime64("2006-01-01T11:45:00")],
            "01/04/2008": [np.datetime64("2008-01-04T07:15:00"),
                           np.datetime64("2008-01-05T06:00:00")],
            "01/07/2017": [np.datetime64("2017-01-07T10:00:00"),
                           np.datetime64("2017-01-12T01:00:00")],
            "01/16/2019": [np.datetime64("2019-01-16T13:15:00"),
                           np.datetime64("2019-01-18T13:00:00")],
            "03/05/2016": [np.datetime64("2016-03-05T09:30:00"),
                           np.datetime64("2016-03-06T20:00:00")],
            "02/12/2019": [np.datetime64("2019-02-13T04:00:00"),
                           np.datetime64("2019-02-17T08:00:00")]
            }

coyote = {"p_csv": "6017P.csv",
          "d_csv": "5012D.csv",
          "03/28/1982": [np.datetime64("1982-03-28T04:15:00"),
                         np.datetime64("1982-04-10T01:45:00")],
          "04/10/1982": [np.datetime64("1982-04-10T08:45:00"),
                         np.datetime64("1982-04-21T08:15:00")],
          "01/26/1983": [np.datetime64("1983-01-26T16:45:00"),
                         np.datetime64("1983-02-05T09:45:00")],
          "02/06/1983": [np.datetime64("1983-02-06T09:45:00"),
                         np.datetime64("1983-02-18T03:00:00")],
          "02/25/1983": [np.datetime64("1983-02-25T10:45:00"),
                         np.datetime64("1983-03-10T13:15:00")],
          "02/15/1986": [np.datetime64("1986-02-15T21:15:00"),
                         np.datetime64("1986-02-23T14:00:00")],
          "03/08/1986": [np.datetime64("1986-03-08T04:00:00"),
                         np.datetime64("1986-03-15T03:30:00")],
          "03/15/1986": [np.datetime64("1986-03-15T10:00:00"),
                         np.datetime64("1986-03-19T20:30:00")],
          "03/09/1995": [np.datetime64("1995-03-09T09:30:00"),
                         np.datetime64("1995-03-22T11:30:00")],
          "03/23/1995": [np.datetime64("1995-03-23T01:45:00"),
                         np.datetime64("1995-03-27T03:00:00")],
          "02/19/1996": [np.datetime64("1996-02-19T13:00:00"),
                         np.datetime64("1996-02-28T06:45:00")],
          "12/31/1996": [np.datetime64("1996-12-31T08:00:00"),
                         np.datetime64("1997-01-07T09:45:00")],
          "01/23/1997": [np.datetime64("1997-01-23T01:45:00"),
                         np.datetime64("1997-02-06T09:30:00")],
          "02/01/1998": [np.datetime64("1998-02-01T12:45:00"),
                         np.datetime64("1998-02-05T04:30:00")],
          "02/06/1998": [np.datetime64("1998-02-06T10:30:00"),
                         np.datetime64("1998-02-12T08:00:00")],
          "02/21/1998": [np.datetime64("1998-02-21T11:00:00"),
                         np.datetime64("1998-03-04T13:45:00")],
          "02/14/1982": [np.datetime64("1982-02-14T13:30:00"),
                         np.datetime64("1982-02-21T23:15:00")],
          "01/23/1983": [np.datetime64("1983-01-23T05:00:00"),
                         np.datetime64("1983-01-26T16:15:00")]
          }


def load_data(precipitation_csv, discharge_csv):
    precipitation_df = pd.read_csv(precipitation_csv,
                                   dtype={'Precipitation': np.float64})
    datetimep = precipitation_df["Datetime"]
    precipitation = precipitation_df["Precipitation"]
    discharge_df = pd.read_csv(discharge_csv,
                               dtype={'Discharge': np.float64},
                               index_col=0)
    datetimed = discharge_df["Datetime"]
    discharge = discharge_df["Discharge"]
    return datetimep, precipitation, datetimed, discharge


def graph_precipitation_discharge(precipitation_csv, discharge_csv, limit):
    precipitation_df = pd.read_csv(precipitation_csv,
                                   dtype={'Precipitation': np.float64},
                                   parse_dates=True)

    discharge_df = pd.read_csv(discharge_csv,
                               dtype={'Discharge': np.float64},
                               parse_dates=True,
                               index_col=0)
    discharge = discharge_df["Discharge"]

    # finding peaks
    peaks, _ = find_peaks(discharge, height=limit, distance=96)
    peak_dates = []
    for i in peaks:
        peak_dates.append(discharge_df["Datetime"].iloc[i])
    print(peak_dates)
    print(len(peak_dates))

    # discharge/precipitation graph
    fig, ax = plt.subplots()
    ax.xaxis_date()
    ax.set_ylabel("Discharge (cfs)")
    ax.plot(discharge_df["mDatetime"], discharge_df["Discharge"], color="blue",
            alpha=0.5)
    ax1 = ax.twinx()
    ax1.plot(precipitation_df["mDatetime"], precipitation_df["Precipitation"],
             color="orange", alpha=0.5)
    ax1.set_ylabel("Precipitation (in)")


def get_event(event, rainfall_before, precipitation_csv, discharge_csv, creek):
    datetimep, precipitation, datetimed, discharge = \
        load_data(precipitation_csv, discharge_csv)
    discharge_start = creek[event][0]
    precipitation_start = discharge_start - np.timedelta64(rainfall_before,
                                                           'm')
    event_end = creek[event][1]
    discharge_start = str(discharge_start)
    precipitation_start = str(precipitation_start)
    event_end = str(event_end)

    p1 = datetimep[datetimep == precipitation_start].index[0]
    p2 = datetimep[datetimep == event_end].index[0]
    event_rainfall = list(precipitation.iloc[p1:p2])
    d1 = datetimed[datetimed == discharge_start].index[0]
    d2 = datetimed[datetimed == event_end].index[0]
    event_discharge = list(discharge.iloc[d1:d2])
    return p1, event_rainfall, event_discharge, precipitation


def get_x_y(event, rainfall_before, hours_before_sum, days_before_rain,
            precipitation_csv, discharge_csv, creek):
    p1, event_rainfall, event_discharge, precipitation = \
        get_event(event, rainfall_before, precipitation_csv, discharge_csv,
                  creek)
    x = []
    y = []
    # turning data into [a, b, c, d], [e] where a, b, c, d are precipitation,
    # e is discharge
    for i in range(len(event_discharge)):
        x_i = []
        for j in range(int(rainfall_before / 15)):
            x_i.append(event_rainfall[i + j])
        hour_before = 0
        for rain in list(precipitation.iloc[p1 - int(hours_before_sum * 4) +
                                            i:p1 + i]):
            hour_before += rain
        x_i.append(hour_before)
        for day in days_before_rain:
            day_before = 0
            for rain in list(precipitation.iloc[p1 - day * 96 + i:p1 + i]):
                day_before += rain
            x_i.append(day_before)
        y_i = event_discharge[i]
        x.append(x_i)
        y.append(y_i)
    x = np.array(x)
    y = np.array(y)
    return x, y


def dense_model(x_train, y_train, l1_l2_reg, prob_1, prob_2, num_epochs):
    model = Sequential()
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(prob_1))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(prob_2))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse',
                  metrics=[keras.metrics.RootMeanSquaredError(),
                           keras.metrics.MeanAbsolutePercentageError()],
                  kernel_regularizer=l1_l2_reg)
    model.fit(x_train, y_train, epochs=num_epochs, verbose=0)
    train_predictions = []
    for index in range(len(x_train)):
        x_input = x_train[index]
        x_input = x_input.reshape((1, len(x_train[0])))
        yhat = model.predict(x_input)
        train_predictions.append(yhat)
    train_predictions = np.array(train_predictions)
    print("Training Metrics:", end=" ")
    print(model.evaluate(x_train, y_train))
    train_predictions = train_predictions.reshape((len(y_train), 1))
    return train_predictions, model


def cnn_model(x_train, y_train):
    model = Sequential()
    model.add(Conv1D(kernel_size=3, filters=5,
                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(Dropout(0.8))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse',
                  metrics=[keras.metrics.RootMeanSquaredError(),
                           keras.metrics.MeanAbsolutePercentageError()])
    model.fit(x_train, y_train, epochs=1000, verbose=0)
    train_predictions = []
    for index in range(len(x_train)):
        x_input = x_train[index]
        x_input = x_input.reshape((1, len(x_train[0])))
        yhat = model.predict(x_input)
        train_predictions.append(yhat)
        print(yhat)
    train_predictions = np.array(train_predictions)
    print("Training Metrics:", end=" ")
    print(model.evaluate(x_train, y_train))
    train_predictions = train_predictions.reshape((len(y_train), 1))
    return train_predictions, model


def graph_prediction(event, event_rainfall, event_discharge, predictions,
                     axis):
    # graphing precipitation
    axis.set_title(event + " Event")
    axis.xaxis.set_tick_params(rotation=30)
    axis.set_ylabel("Precipitation (in)")
    event_p = axis.bar(x=range(len(event_rainfall)),
                       height=-np.array(event_rainfall),
                       width=0.5,
                       color="orange")
    yp_ticks = [0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4]
    yp_ticklabels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    axis.set_yticks(yp_ticks)
    axis.set_yticklabels(yp_ticklabels)

    # graphing discharge
    axd = axis.twinx()
    event_d = axd.plot(range(len(event_discharge)),
                       event_discharge,
                       color="blue")
    axd.set_ylabel("Discharge (cfs)")

    # graphing predictions
    axp = axis.twinx()
    prediction_graph = axp.plot(range(len(event_discharge)),
                                predictions, color="red")
    axp.axes.get_yaxis().set_visible(False)
    return event_p, event_d, prediction_graph


def predict_event(model, x_test, y_test):
    test_predictions = model.predict(x_test, verbose=0)
    print("Testing Metrics:", end=" ")
    print(model.evaluate(x_test, y_test))
    return test_predictions


def train_test_split_all(num_events, creek, test, num_val_events,
                         num_test_events):
    if test:
        all_indices = list(range(num_events))
        train_events_indices = all_indices[2:num_events - num_test_events]
        test_events_indices = all_indices[num_events - num_test_events:
                                          num_events]
        train_events = [list(creek.keys())[i] for i in train_events_indices]
        test_events = [list(creek.keys())[i] for i in test_events_indices]
        return train_events, test_events
    else:
        all_indices = list(range(num_events))
        train_events_indices = all_indices[2:num_events - num_test_events -
                                           num_val_events]
        test_events_indices = all_indices[num_events - num_val_events -
                                          num_test_events:
                                          num_events - num_test_events]
        train_events = [list(creek.keys())[i] for i in train_events_indices]
        test_events = [list(creek.keys())[i] for i in test_events_indices]
        return train_events, test_events


def combine_events(hours_before, hours_before_sum, events, days_before_rain,
                   precipitation_csv, discharge_csv, creek):
    time_before = int(hours_before * 60)
    all_event_rainfall = np.array([])
    all_event_discharge = np.array([])
    x = np.array([])
    y = np.array([])
    for event_i in events:
        p1_i, event_rainfall_i, event_discharge_i, precipitation = \
            get_event(event_i, time_before, precipitation_csv, discharge_csv,
                      creek)
        xi, yi = get_x_y(event_i, time_before, hours_before_sum,
                         days_before_rain, precipitation_csv, discharge_csv,
                         creek)
        all_event_rainfall = np.concatenate((all_event_rainfall,
                                             event_rainfall_i))
        all_event_discharge = np.concatenate((all_event_discharge,
                                              event_discharge_i))
        x = np.concatenate((x, xi.flatten()))
        y = np.concatenate((y, yi.flatten()))
    return all_event_rainfall, all_event_discharge, x, y


def train_predict(hours_before_non_sum, hours_before_sum, days_before_rain,
                  regularizer, prob_1, prob_2, num_epochs,
                  precipitation_csv, discharge_csv, creek,
                  test, num_val_events, num_test_events):
    start_time = time.time()
    title = "" + str(hours_before_non_sum) + ", " + "" + \
            str(hours_before_sum) + ", " + str(days_before_rain) + ", " + \
            str(type(regularizer)) + "," + str(prob_1) + ", " + \
            str(prob_2) + ", " + str(num_epochs)
    print(title)
    time_before_non_sum = int(hours_before_non_sum * 60)
    all_train_events, all_test_events = \
        train_test_split_all(len(creek), creek, test, num_val_events,
                             num_test_events)
    all_train_event_rainfall, all_train_event_discharge, all_x_train, \
        all_y_train = combine_events(time_before_non_sum, hours_before_sum,
                                     all_train_events, days_before_rain,
                                     precipitation_csv, discharge_csv, creek)
    all_test_event_rainfall, all_test_event_discharge, all_x_test, \
        all_y_test = combine_events(time_before_non_sum, hours_before_sum,
                                    all_test_events, days_before_rain,
                                    precipitation_csv, discharge_csv, creek)
    all_x_train = all_x_train.reshape(int(len(all_x_train) /
                                          (time_before_non_sum * 4 +
                                           len(days_before_rain) + 1)),
                                      int(time_before_non_sum * 4 +
                                          len(days_before_rain)) + 1)
    all_x_test = all_x_test.reshape(int(len(all_x_test) /
                                        (time_before_non_sum * 4 +
                                         len(days_before_rain) + 1)),
                                    int(time_before_non_sum * 4 +
                                        len(days_before_rain)) + 1)

    fig_train, ax_train = plt.subplots()
    fig_test, ax_test = plt.subplots()
    all_train_predictions, all_model = dense_model(all_x_train, all_y_train,
                                                   regularizer, prob_1, prob_2,
                                                   num_epochs)
    all_test_predictions = predict_event(all_model, all_x_test, all_y_test)
    graph_prediction(title, all_train_event_rainfall,
                     all_train_event_discharge,
                     all_train_predictions, ax_train)
    graph_prediction(title, all_test_event_rainfall, all_test_event_discharge,
                     all_test_predictions, ax_test)
    print("--- %s seconds ---" % (time.time() - start_time))
    return fig_train, ax_train, fig_test, ax_test


train_predict(hours_before_non_sum=6.5, hours_before_sum=6.5,
              days_before_rain=list(range(1, 8, 2)), regularizer=None,
              prob_1=1, prob_2=1, num_epochs=1000,
              precipitation_csv=coyote["p_csv"],
              discharge_csv=coyote["d_csv"],
              creek=coyote, test=False, num_val_events=1,
              num_test_events=1)
train_predict(hours_before_non_sum=7, hours_before_sum=7,
              days_before_rain=list(range(1, 8, 2)), regularizer=None,
              prob_1=1, prob_2=1, num_epochs=1000,
              precipitation_csv=coyote["p_csv"],
              discharge_csv=coyote["d_csv"],
              creek=coyote, test=False, num_val_events=1,
              num_test_events=1)

plt.show()
