import pandas as pd
import datetime as dt
import matplotlib.dates as mdates

df = pd.read_csv("6017P.csv")
# df["Datetime"] = df["Date"] + df["Time"]
# new_dt = []
new_mdt = []
# datetimeobject = dt.datetime.strptime(string, '%m/%d/%Y%H:%M:%S')
#     new_dt.append(datetimeobject.strftime('%Y-%m-%dT%H:%M:%S'))
for string in df["Datetime"]:

    new_mdt.append(mdates.datestr2num(string))
# df["Datetime"] = new_dt
df["mDatetime"] = new_mdt
print(df.head())
df.to_csv("6017P.csv")
