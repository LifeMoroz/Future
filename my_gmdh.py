import datetime as dt

import pandas as pd
from gmdhpy.gmdh import Regressor
import matplotlib.pyplot as plt

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
]

test = [
    58324,
    61798,
    38664,
    54106,
]


def resample_data(raw):
    data = [i * 1. / max(raw) for i in raw]

    start = dt.datetime.strptime("1 Nov 15", "%d %b %y")
    daterange = pd.date_range(start, periods=len(data))
    table = {"count": data, "date": daterange}
    data = pd.DataFrame(table)
    data.set_index("date", inplace=True)
    data = data.resample('2D').mean()

    return list(x[0] for x in data.values)


data = resample_data(raw)


def slicing(raw, size):
    i = 0
    while len(raw) > i + size:
        yield raw[i:i + size], raw[i + size]
        i += 1


train_x = []
train_y = []
SIZE = 5
TEST_OFFSET = -SIZE - 1
for x, y in slicing(data, SIZE):
    train_x.append(x)
    train_y.append((y,))

model = Regressor()
model.fit(train_x, train_y)


predicted = []
for x in range(4):
    predict_y = model.predict([data[TEST_OFFSET - x:TEST_OFFSET - x + SIZE]])
    predicted.append(predict_y[0])

for x in range(1, 5):
    predicted.append(model.predict([(data + predicted)[-SIZE:]])[0])

predicted = [p * max(raw) for p in predicted]

print predicted
raw += test
plt.plot(range(len(raw)), list(raw))
plt.plot(range(len(raw) - len(test) - 1, len(raw) - len(test) + len(predicted) - 1), predicted)
plt.show()
