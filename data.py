import nemos.glm
import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt
import matplotlib
import pynapple as nap
import nemos as nmo

matplotlib.use("TkAgg")

song_times = sio.loadmat('c57SongTimes.mat')['c57SongTimes']
audio_segm = sio.loadmat('c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('c57LightOffTime.mat')['c57LightOffTime']

spikes_all = sio.loadmat('c57SpikeTimesAll.mat')['c57SpikeTimesAll']
spikes_quiet = sio.loadmat('c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']

ts_dict = {key: nap.Ts(spikes_all[key, 0].flatten()) for key in range(spikes_all.shape[0])}

spike_times = nap.TsGroup(ts_dict)

# add metadata
spike_times["neuron_type"] = [0] * 185 + [1] * 10

# count a subpopulation
binsize = 0.1   # in seconds
counts = spike_times[spike_times.neuron_type == 1].count(binsize)

# define  basis
basis = nmo.basis.RaisedCosineBasisLog(3, mode="conv", window_size=10)

# plot basis
plt.plot(basis(np.linspace(0, 1, 10)))

# apply basis to spikes
X = basis.compute_features(counts)

# model
model = nmo.glm.PopulationGLM()
model.fit(X, counts)