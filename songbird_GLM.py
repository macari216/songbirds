import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
import pynapple as nap
import nemos as nmo
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.utils import gen_batches

matplotlib.use("TkAgg")

song_times = sio.loadmat('/Users/macari216/Desktop/songbirds/songbirds/c57SongTimes.mat')['c57SongTimes']
audio_segm = sio.loadmat('/Users/macari216/Desktop/songbirds/songbirds/c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('/Users/macari216/Desktop/songbirds/songbirds/c57LightOffTime.mat')['c57LightOffTime']
spikes_all = sio.loadmat('/Users/macari216/Desktop/songbirds/songbirds/c57SpikeTimesAll.mat')['c57SpikeTimesAll']
spikes_quiet = sio.loadmat('/Users/macari216/Desktop/songbirds/songbirds/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']

#convert times to Interval Sets and spikes to TsGroups
song_times = nap.IntervalSet(start=song_times[:,0], end=song_times[:,1])
audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

ts_dict_all = {key: nap.Ts(spikes_all[key, 0].flatten()) for key in range(spikes_all.shape[0])}
spike_times = nap.TsGroup(ts_dict_all)

ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

# add neuron subset marker
spike_times["neuron_subset"] = [0] * 95 + [1] * 100

# # create batches
# batches = gen_batches(spike_times.time_support, 10)

# count a subpopulation during the first 5 minutes
binsize = 0.01   # in seconds
# count = spike_times[spike_times.neuron_type == 1].count(binsize, ep=nap.IntervalSet(0,180))
# count = spike_times.count(binsize, ep=nap.IntervalSet(0,off_time))
count = spike_times.count(binsize, ep=nap.IntervalSet(0, 5*60))
count = nap.TsdFrame(t=count.t, d=count.values)

#spike count series (1 neuron)
plot_series = nap.IntervalSet(start=song_times.start[3], end=song_times.start[3]+30)
# plt.figure(figsize=(7, 3.5))
# plt.step(count[:,0].restrict(song_times[0]).t, count[:,0].restrict(song_times[0]).d, where="post")
# plt.xlabel("Time (sec)")
# plt.ylabel("Spikes")


# SEARCH FOR BEST BASIS PARAMS
hist_window_sec = np.arange(0.1, 3, 0.1)
hist_window_size = [int(hist_window_sec[i] * count.rate) for i in range(len(hist_window_sec))]
basis_fun = np.arange(2,11)

pipe = Pipeline([('basis', nmo.basis.RaisedCosineBasisLog(2,mode="conv",window_size=10)),
                 ('glm', nmo.glm.PopulationGLM(regularizer=nmo.regularizer.Ridge(regularizer_strength=0.1, solver_name="LBFGS")))])

pipe.fit(count)

quit()

param_grid  = {
    "basis__n_basis_funcs":basis_fun,
    "basis__window_size": hist_window_size,
}

clf = GridSearchCV(pipe, param_grid)
clf.fit(count)

print(clf.best_params_)

#choose spike history window
hist_window_sec = 0.9
hist_window_size = int(hist_window_sec * count.rate)

# define  basis
basis = nmo.basis.RaisedCosineBasisLog(6, mode="conv", window_size=hist_window_size)
time, basis_kernels = basis.evaluate_on_grid(hist_window_size)

# plot basis
# plt.figure(figsize=(7, 3))
# plt.plot(basis(np.linspace(0, 1, 100)))

# apply basis to spikes
X = basis.compute_features(count)

# train test split 60-40
duration = X.time_support.tot_length("s")
start = X.time_support["start"]
end = X.time_support["end"]
training = nap.IntervalSet(start, start + duration * 0.6)
testing = nap.IntervalSet(start + duration * 0.6, end)

# model
model = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.Ridge(regularizer_strength=0.1, solver_name="LBFGS"))
model.fit(X.restrict(training), count.restrict(training))

# compute score
score_train = model.score(X.restrict(training), count.restrict(training), score_type="pseudo-r2-McFadden")
score_test = model.score(X.restrict(testing), count.restrict(testing), score_type="pseudo-r2-McFadden")
print("Score(train data):", score_train)
print("Score(test data):", score_test)

# coupling weights
weights = model.coef_.reshape(count.shape[1], basis.n_basis_funcs, count.shape[1])
responses = np.einsum("jki,tk->ijt", weights, basis_kernels)

# plot coupling weights for 2 neurons
# plt.figure()
# plt.title("Coupling Filter Between Neurons 1 and 2")
# plt.plot(time, responses[0,1,:], "k", lw=2)
# plt.axhline(0, color="k", lw=0.5)
# plt.xlabel("Time from spike (sec)")
# plt.ylabel("Weight")

# calculate spike predictions
spike_pred = model.predict(X) * count.rate

# plot predicted firing rates
# plt.figure(figsize=(8,2))
# firing_rate = spike_pred.restrict(plot_series).d[:,:15]
# firing_rate = firing_rate.T / np.max(firing_rate, axis=1)
# plt.imshow(firing_rate[::-1], cmap="Blues", aspect="auto")
# plt.ylabel("Neuron")
# plt.xlabel("Time (sec)")
# plt.show()

# plot neural activity
# spikes_sub = count.restrict(plot_series)[:, :15]
# fig = plt.figure(figsize=(8,2))
# ax = plt.subplot2grid(
#     (4, 15), loc=(1, 0), rowspan=1, colspan=15, fig=fig)
# for i in range(15):
#     sel = spikes_sub.d[:,i]
#     spikes_tsd = nap.Tsd(t=spikes_sub.t, d=sel)
#     ax.plot(
#         spikes_tsd.t,
#         np.ones(sel.sum()) * i,
#         "|")
# plt.show()

