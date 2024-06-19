import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pynapple as nap
import nemos as nmo
from sklearn.utils import gen_batches

song_times = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57SongTimes.mat')['c57SongTimes']
audio_segm = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57LightOffTime.mat')['c57LightOffTime']
spikes_quiet = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
spikes_all = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57SpikeTimesAll.mat')['c57SpikeTimesAll']

#convert times to Interval Sets and spikes to TsGroups
audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

audio_on = audio_segm[:149]
time_on = nap.IntervalSet(0, off_time)
time_quiet = time_on.set_diff(audio_on)
time_quiet = time_quiet.drop_short_intervals(1,'s')

ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

# add neuron subset marker
spike_times_quiet["neuron_subset"] = [0] * 195 + [1] * 0

# spike count
binsize = 0.1   # in seconds
# count = spike_times_quiet[spike_times_quiet.neuron_subset == 0].count(binsize, ep=time_quiet)
count = spike_times_quiet.count(binsize, ep=time_quiet)

#choose spike history window
hist_window_sec = 0.9
hist_window_size = int(hist_window_sec * count.rate)

# define  basis
basis = nmo.basis.RaisedCosineBasisLog(9, mode="conv", window_size=hist_window_size)
time, basis_kernels = basis.evaluate_on_grid(hist_window_size)
time *= hist_window_sec

X = basis.compute_features(count)

# train test split 70-30
duration = count.time_support.tot_length("s")
start = count.time_support["start"]
end = count.time_support["end"]
training = nap.IntervalSet(start[0], start[0] + duration * 0.7)
testing = nap.IntervalSet(start[0] + duration * 0.7, end[-1])

# define and train model
model = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.Ridge(regularizer_strength=1.0, solver_name="LBFGS"))
model.fit(X.restrict(training), count.restrict(training).squeeze())

# compute score
score_train = model.score(X.restrict(training), count.restrict(training).squeeze(), score_type="pseudo-r2-McFadden")
score_test = model.score(X.restrict(testing), count.restrict(testing).squeeze(), score_type="pseudo-r2-McFadden")
print("Score(train data):", score_train)
print("Score(test data):", score_test)

# model output
weights = model.coef_.reshape(count.shape[1], basis.n_basis_funcs, count.shape[1])
filters = np.einsum("jki,tk->ijt", weights, basis_kernels)
spike_pred = model.predict(X.restrict(testing))
intercept = model.intercept_
coef = model.coef_
results_dict = {"weights": weights, "filters": filters, "intercept": intercept, "coef": coef,
                "spike_pred": spike_pred, "time": time}

np.save("results.npy", results_dict)
