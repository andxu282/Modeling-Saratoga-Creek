import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks

scaler = StandardScaler()

precipitation_df = pd.read_csv("6077P.csv",
                               dtype={'Precipitation': np.float64},
                               parse_dates=True)
precipitation = precipitation_df["Precipitation"]
# precipitation graph
# figp, axp = plt.subplots()
# axp.xaxis_date()
# axp.plot(precipitation_df["datetime"], precipitation_df["precipitation"],
#           color="orange")

discharge_df = pd.read_csv("5096D.csv",
                           dtype={'Discharge': np.float64},
                           parse_dates=True,
                           index_col=0)
discharge = discharge_df["Discharge"]

# discharge graph
# figd, axd = plt.subplots()
# axd.xaxis_date()
# axd.plot(discharge_df["datetime"], discharge, color="blue")
# axd.plot(discharge_df["datetime"], moving_average, color="red")

# finding peaks
peaks, _ = find_peaks(discharge, height=800, distance=96)
peak_dates = []
for i in peaks:
    peak_dates.append(discharge_df["Datetime"].iloc[i])
print(peak_dates)
print(len(peak_dates))

# discharge/precipitation graph
datetime = discharge_df["Datetime"]
fig, ax = plt.subplots()
# x = []
# y = []
# for i in peak_dates:
#     id = datetime[datetime == i].index[0]
#     x.append(discharge_df["datetime"][id])
#     y.append(discharge[id])
# ax.plot(x, y, color="black")
ax.xaxis_date()
ax.set_ylabel("Discharge (cfs)")
ax.plot(discharge_df["mDatetime"], discharge_df["Discharge"], color="blue",
        alpha=0.5)
ax1 = ax.twinx()
ax1.plot(precipitation_df["mDatetime"], precipitation_df["Precipitation"],
         color="orange", alpha=0.5)
ax1.set_ylabel("Precipitation (in)")
plt.show()
