import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
import pynapple as nap
import nemos as nmo
from sklearn.utils import gen_batches
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


matplotlib.use("TkAgg")

song_times = sio.loadmat('/Users/macari216/Desktop/glm-songbirds/songbirds/c57SongTimes.mat')['c57SongTimes']
audio_segm = sio.loadmat('/Users/macari216/Desktop/glm-songbirds/songbirds/c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('/Users/macari216/Desktop/glm-songbirds/songbirds/c57LightOffTime.mat')['c57LightOffTime']
spikes_all = sio.loadmat('/Users/macari216/Desktop/glm-songbirds/songbirds/c57SpikeTimesAll.mat')['c57SpikeTimesAll']
spikes_quiet = sio.loadmat('/Users/macari216/Desktop/glm-songbirds/songbirds/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']

#convert times to Interval Sets and spikes to TsGroups
song_times = nap.IntervalSet(start=song_times[:,0], end=song_times[:,1])
audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

# add neuron subset marker
spike_times_quiet["neuron_subset"] = [0] * 10 + [1] * 185

# spike count
binsize = 0.01   # in seconds
count = spike_times_quiet.count(binsize, ep=nap.IntervalSet(0, off_time))

#choose spike history window
hist_window_sec = 0.5
hist_window_size = int(hist_window_sec * count.rate)

# SEARCH FOR BEST BASIS PARAMS
# hist_window_sec = np.arange(0.1, 3, 0.1)
# hist_window_size = [int(hist_window_sec[i] * count.rate) for i in range(len(hist_window_sec))]
basis_fun = np.arange(2,11)

basis = nmo.basis.RaisedCosineBasisLog(6, mode="conv", window_size=hist_window_size)
bas = nmo.basis.TransformerBasis(basis**195)
model = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.Ridge(regularizer_strength=0.1, solver_name="LBFGS"))
pipe = Pipeline([("eval", bas), ("fit", model)])

param_grid  = {"eval__basis":[nmo.basis.RaisedCosineBasisLog(3, mode="conv", window_size=50)**195,
                              nmo.basis.RaisedCosineBasisLog(3, mode="conv", window_size=90)**195,
                              nmo.basis.RaisedCosineBasisLog(6, mode="conv", window_size=50)**195,
                              nmo.basis.RaisedCosineBasisLog(6, mode="conv", window_size=90)**195,
                              nmo.basis.RaisedCosineBasisLog(9, mode="conv", window_size=50)**195,
                              nmo.basis.RaisedCosineBasisLog(9, mode="conv", window_size=90)**195]}

cv = GridSearchCV(pipe, param_grid)
cv.fit(count, count.squeeze())

res = cv.best_params_
print(res['eval__basis'].n_basis_funcsn)