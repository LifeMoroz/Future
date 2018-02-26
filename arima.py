import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

raw = [
    43550,
    42184,
    41026,
    27658,
    39310,
    49146,
    83450,
    72534,
    58018,
    61968,
    53016,
    62206,
    50430,
    54408,
    57964,
    37160,
    49040,
    56826,
    61944,
    65102,
    56102,
    60450,
    63808,
    65808,
    68546,
    59644,
    58324,
]
data = [i * 1. / max(raw) for i in raw]

start = dt.datetime.strptime("1 Nov 15", "%d %b %y")
daterange = pd.date_range(start, periods=len(data))
table = {"count": data, "date": daterange}
data = pd.DataFrame(table)
data.set_index("date", inplace=True)
data = data.resample('2D', how='mean')


order = (2, 1, 1)
model = ARIMA(data, order)
model = model.fit()

y = []

for a in model.predict(1, 5, typ='levels'):
    y.append(a * max(raw))

new_x = list(pd.date_range(start + dt.timedelta(days=len(raw)), periods=len(y)))

raw += [
    61798,
    38664,
    54106,
]

plt.plot(list(pd.date_range(start, periods=len(raw))), list(raw))
plt.plot(new_x, y)
plt.show()
