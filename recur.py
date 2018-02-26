from itertools import cycle

from pybrain.datasets import SequentialDataSet
from pybrain.structure.modules.lstm import LSTMLayer
from pybrain.supervised import RPropMinusTrainer
from pybrain.tools.shortcuts import buildNetwork

ds = SequentialDataSet(1, 1)
data = [
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
    68546,
    59644,
    58324,
    61798,
    38664,
    54106,
]


m = max(data) * 1.
for sample, next_sample in zip(data, cycle(data[1:])):
    ds.addSample(sample/m, next_sample/m)

# construct LSTM network - note the missing output bias
net = buildNetwork(1, 20, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)

trainer = RPropMinusTrainer(net, dataset=ds)
train_errors = []  # save errors for plotting later
EPOCHS_PER_CYCLE = 5
CYCLES = 100
EPOCHS = EPOCHS_PER_CYCLE * CYCLES
for i in xrange(CYCLES):
    trainer.trainEpochs(EPOCHS_PER_CYCLE)
    train_errors.append(trainer.testOnData())
    epoch = (i + 1) * EPOCHS_PER_CYCLE
    print("\r epoch {}/{}".format(epoch, EPOCHS))

print("final error =", train_errors[-1])
# plt.plot(range(0, EPOCHS, EPOCHS_PER_CYCLE), train_errors)
# plt.xlabel('epoch')
# plt.ylabel('error')
# plt.show()

for sample, target in ((54106, 53222), (53222, 55482), (55482, 63612), (63612, 55264)):
    print("               sample = %4.1f" % sample)
    print("predicted next sample = %4.1f" % (net.activate(sample/m) * m))
    print("   actual next sample = %4.1f" % (target))
    print("delta: %4.1f" % (net.activate(sample/m) * m - target) )