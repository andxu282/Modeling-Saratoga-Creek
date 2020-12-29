import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# TODO:
# - make graphs prettier
# - produce C values for the other nine events √
# - optimize taking out the data indices √
# - add up rainfall before and see the effect on the C value √
# - bar graphs for precipitation in lieu of line graph √
# - show points for p1, p2, d1, d2 √
# - graph week before p1 and day after d2 √
# - 2 graphs, 1 that is zoomed into event, other has week before √
# - 1, 3, 5 day total rainfall before p1 √
# - use scipy.signal.find_peaks, then see if any of the peaks are within a day
#   of each other, take only one of them √
# For Deep Learning Model:
# - Change the data into an image
# - Use a CNN model w/Transfer Learning (idk if transfer learning is possible)
#   since I only have one gauge

precipitation_df = pd.read_csv("precipitation_15_min_new_dt.csv",
                               dtype={'precipitation': np.float64})
precipitation_mdates = pd.read_csv("precipitation_15_min_mdates.csv",
                                   dtype={'precipitation': np.float64},
                                   parse_dates=True)
datetimep = precipitation_df["datetime"]
print(precipitation_mdates)
discharge_df = pd.read_csv("discharge_15_min_new_dt.csv",
                           dtype={'discharge': np.float64},
                           index_col=0)
discharge_mdates = pd.read_csv("discharge_15_min_mdates.csv",
                               dtype={'discharge': np.float64},
                               index_col=0,
                               parse_dates=True)
datetimed = discharge_df["datetime"]


def to_image():
    print("yes")


def find_c(p_start, p_end, d_start, d_end, base_flow):
    print(p_start[6:10])
    p1 = datetimep[datetimep == p_start].index[0]
    p2 = datetimep[datetimep == p_end].index[0]
    print(p1)
    print(p2)
    event_rainfall = list(precipitation_df["precipitation"].iloc[p1:p2])

    five_before = 0
    for i in list(precipitation_df["precipitation"].iloc[p1 - 480:p1]):
        five_before += i
    five_before = round(five_before, 2)
    print("Five Day Before Precipitation:")
    print(str(five_before) + " inches")

    three_before = 0
    for i in list(precipitation_df["precipitation"].iloc[p1 - 288:p1]):
        three_before += i
    three_before = round(three_before, 2)
    print("Three Day Before Precipitation:")
    print(str(three_before) + " inches")

    one_before = 0
    for i in list(precipitation_df["precipitation"].iloc[p1 - 96:p1]):
        one_before += i
    one_before = round(one_before, 2)
    print("One Day Before Precipitation:")
    print(str(one_before) + " inches")

    total_rainfall = 0
    for i in event_rainfall:
        total_rainfall += i
    #    print(str(total_rainfall) + " inches")
    total_rainfall = total_rainfall / 12 * 9.22 * 640
    total_rainfall = round(total_rainfall, 2)
    print("Event Precipitation: ")
    # print(str(total_rainfall) + " acre-foot")

    d1 = datetimed[datetimed == d_start].index[0]
    d2 = datetimed[datetimed == d_end].index[0]
    event = list(discharge_df["discharge"].iloc[d1:d2])
    event_discharge = 0
    for i in range(len(event) - 1):
        event_discharge += (event[i] + event[i + 1] - 2 * base_flow) * 15 * 30
    #    print(str(event_discharge) + " cubic feet")
    event_discharge /= 43560
    print("Event Discharge: ")
    print(str(event_discharge) + " acre-foot")
    c_value = event_discharge / total_rainfall
    print(c_value)
    print()
    return five_before, three_before, one_before, c_value


def graph_event(p_start, p_end, d_start, d_end, axp, name, show_week_before):
    if show_week_before:
        p1 = datetimep[datetimep == p_start].index[0] - 672
        p2 = datetimep[datetimep == p_end].index[0] + 96
        d1 = datetimed[datetimed == d_start].index[0] - 672
        d2 = datetimed[datetimed == d_end].index[0] + 96
    else:
        p1 = datetimep[datetimep == p_start].index[0]
        p2 = datetimep[datetimep == p_end].index[0]
        d1 = datetimed[datetimed == d_start].index[0]
        d2 = datetimed[datetimed == d_end].index[0]
    event_rainfall = precipitation_mdates["precipitation"].iloc[p1:p2]
    event = discharge_mdates["discharge"].iloc[d1:d2]
    axp.set_title(name + " Event")
    axp.xaxis.set_tick_params(rotation=30)
    axp.xaxis_date()
    axp.set_ylabel("Precipitation (in)")
    event_p = axp.bar(x=precipitation_mdates["datetime"].iloc[p1:p2],
                      height=-1 * event_rainfall,
                      width=12. / 24 / 60,
                      color="orange")
    yp_ticks = [0, -0.05, -0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4]
    yp_ticklabels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    axp.set_yticks(yp_ticks)
    axp.set_yticklabels(yp_ticklabels)
    axd = axp.twinx()
    event_d = axd.plot(discharge_mdates["datetime"].iloc[d1:d2],
                       event,
                       color="blue")
    axd.set_ylabel("Discharge (cfs)")
    return event_p, event_d


# fig, ax = plt.subplots()
# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
# fig3, ax3 = plt.subplots()
# fig4, ax4 = plt.subplots()
# fig5, ax5 = plt.subplots()
# fig6, ax6 = plt.subplots()
# fig7, ax7 = plt.subplots()
# fig8, ax8 = plt.subplots()
# fig9, ax9 = plt.subplots()
# fig10, ax10 = plt.subplots()
# graph_event("02/02/1998 11:45:00", "02/03/1998 13:15:00",
#             "02/02/1998 10:45:00", "02/03/1998 20:15:00", ax,
#             "February 3rd, 1998", True)
# graph_event("01/16/2019 11:30:00", "01/17/2019 16:15:00",
#             "01/16/2019 13:15:00", "01/18/2019 13:00:00", ax1,
#             "January 16th, 2019", True)
# graph_event("12/15/2002 00:45:00", "12/16/2002 06:15:00",
#             "12/16/2002 01:00:00", "12/16/2002 18:45:00", ax2,
#             "December 16th, 2002", True)
# graph_event("12/30/2005 10:15:00", "12/31/2005 12:15:00",
#             "12/30/2005 18:00:00", "01/01/2006 11:45:00", ax3,
#             "December 30th, 2005", True)
# graph_event("01/10/2017 02:45:00", "01/11/2017 05:30:00",
#             "01/10/2017 04:30:00", "01/12/2017 01:00:00", ax4,
#             "January 10th, 2017", True)
# graph_event("02/12/2019 21:15:00", "02/17/2019 06:45:00",
#             "02/13/2019 04:00:00", "02/17/2019 08:00:00", ax5,
#             "February 13th, 2019", True)
# graph_event("12/29/1996 04:45:00", "01/05/1997 01:00:00",
#             "12/29/1996 09:45:00", "01/07/1997 12:30:00", ax6,
#             "December 29th, 1996", True)
# graph_event("01/04/2008 01:00:00", "01/04/2008 19:00:00",
#             "01/04/2008 02:00:00", "01/05/2008 06:00:00", ax7,
#             "January 4th, 2008", True)
# graph_event("02/07/1998 02:15:00", "02/08/1998 16:45:00",
#             "02/07/1998 08:30:00", "02/09/1998 05:00:00", ax8,
#             "February 7th, 1998", True)
# graph_event("01/07/2017 02:45:00", "01/09/2017 03:45:00",
#             "01/07/2017 10:00:00", "01/09/2017 15:15:00", ax9,
#             "January 7th, 2017", True)
# graph_event("03/05/2016 07:00:00", "03/06/2016 07:00:00",
#             "03/05/2016 09:30:00", "03/06/2016 20:00:00", ax10,
#             "March 5th, 2016", True)
# plt.show()
five_day = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
three_day = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
one_day = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
five_day[0], three_day[0], one_day[0], c[0] = find_c("02/02/1998 11:45:00",
                                                     "02/03/1998 13:15:00",
                                                     "02/02/1998 10:45:00",
                                                     "02/03/1998 20:15:00",
                                                     125)
five_day[1], three_day[1], one_day[1], c[1] = find_c("01/16/2019 11:30:00",
                                                     "01/17/2019 16:15:00",
                                                     "01/16/2019 13:15:00",
                                                     "01/18/2019 13:00:00", 50)
five_day[2], three_day[2], one_day[2], c[2] = find_c("12/15/2002 00:45:00",
                                                     "12/16/2002 06:15:00",
                                                     "12/16/2002 01:00:00",
                                                     "12/16/2002 18:45:00", 50)
five_day[3], three_day[3], one_day[3], c[3] = find_c("12/30/2005 10:15:00",
                                                     "12/31/2005 12:15:00",
                                                     "12/30/2005 18:00:00",
                                                     "01/01/2006 11:45:00", 55)
five_day[4], three_day[4], one_day[4], c[4] = find_c("01/10/2017 02:45:00",
                                                     "01/11/2017 05:30:00",
                                                     "01/10/2017 04:30:00",
                                                     "01/12/2017 01:00:00",
                                                     120)
five_day[5], three_day[5], one_day[5], c[5] = find_c("02/12/2019 21:15:00",
                                                     "02/17/2019 06:45:00",
                                                     "02/13/2019 04:00:00",
                                                     "02/17/2019 08:00:00", 40)
five_day[6], three_day[6], one_day[6], c[6] = find_c("12/29/1996 04:45:00",
                                                     "01/05/1997 01:00:00",
                                                     "12/29/1996 09:45:00",
                                                     "01/07/1997 12:30:00",
                                                     150)
five_day[7], three_day[7], one_day[7], c[7] = find_c("01/04/2008 01:00:00",
                                                     "01/04/2008 19:00:00",
                                                     "01/04/2008 02:00:00",
                                                     "01/05/2008 06:00:00", 3)
five_day[8], three_day[8], one_day[8], c[8] = find_c("02/07/1998 02:15:00",
                                                     "02/08/1998 16:45:00",
                                                     "02/07/1998 08:30:00",
                                                     "02/09/1998 05:00:00",
                                                     200)
five_day[9], three_day[9], one_day[9], c[9] = find_c("01/07/2017 02:45:00",
                                                     "01/09/2017 03:45:00",
                                                     "01/07/2017 10:00:00",
                                                     "01/09/2017 15:15:00", 32)
five_day[10], three_day[10], one_day[10], c[10] = find_c("03/05/2016 07:00:00",
                                                         "03/06/2016 07:00:00",
                                                         "03/05/2016 09:30:00",
                                                         "03/06/2016 20:00:00",
                                                         3)
# five_day.sort()
# three_day.sort()
# one_day.sort()
# fig, ax = plt.subplots()
# fig1, ax1 = plt.subplots()
# fig2, ax2 = plt.subplots()
# ax.plot(five_day, c)
# ax1.plot(three_day, c)
# ax2.plot(one_day, c)
plt.show()
