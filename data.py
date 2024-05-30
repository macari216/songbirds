import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
import pynapple as nap
import nemos as nmo

matplotlib.use("TkAgg")

song_times = sio.loadmat('c57SongTimes.mat')['c57SongTimes']
audio_segm = sio.loadmat('c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('c57LightOffTime.mat')['c57LightOffTime']

#convert times to Interval Sets
song_times = nap.IntervalSet(start=song_times[:,0], end=song_times[:,1])
audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

spikes_all = sio.loadmat('c57SpikeTimesAll.mat')['c57SpikeTimesAll']
spikes_quiet = sio.loadmat('c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']

ts_dict = {key: nap.Ts(spikes_all[key, 0].flatten()) for key in range(spikes_all.shape[0])}
spike_times = nap.TsGroup(ts_dict)

# add neuron type marker
spike_times["neuron_type"] = [0] * 95 + [1] * 100

# count a subpopulation during the first 3 minutes
binsize = 0.01   # in seconds
# count = spike_times[spike_times.neuron_type == 1].count(binsize, ep=nap.IntervalSet(0,180))
count = spike_times.count(binsize, ep=nap.IntervalSet(0,5*60))
count = nap.TsdFrame(t=count.t, d=count.values)

#spike count series (1 neuron)
plot_series = nap.IntervalSet(start=song_times.start[3], end=song_times.start[3]+30)
# plt.figure(figsize=(7, 3.5))
# plt.step(count[:,0].restrict(song_times[0]).t, count[:,0].restrict(song_times[0]).d, where="post")
# plt.xlabel("Time (sec)")
# plt.ylabel("Spikes")

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
testing = nap.IntervalSet(start + duration * 0.4, end)

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

# plt.figure()
# plt.title("Coupling Filter Between Neurons 1 and 2")
# plt.plot(time, responses[0,1,:], "k", lw=2)
# plt.axhline(0, color="k", lw=0.5)
# plt.xlabel("Time from spike (sec)")
# plt.ylabel("Weight")

# predictions
spike_pred = model.predict(X) * count.rate

plt.figure(figsize=(8,2))
firing_rate = spike_pred.restrict(plot_series).d[:,:15]
firing_rate = firing_rate.T / np.max(firing_rate, axis=1)
plt.imshow(firing_rate[::-1], cmap="Blues", aspect="auto")
plt.ylabel("Neuron")
plt.xlabel("Time (sec)")
plt.show()

spikes_tsd = spike_times.restrict(plot_series).to_tsd()

