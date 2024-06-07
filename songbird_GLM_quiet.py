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

ts_dict_all = {key: nap.Ts(spikes_all[key, 0].flatten()) for key in range(spikes_all.shape[0])}
spike_times_all = nap.TsGroup(ts_dict_all)

ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

# add neuron subset marker
spike_times_all["neuron_subset"] = [0] * 10 + [1] * 185

# spike count
binsize = 0.01   # in seconds
count = spike_times_all.count(binsize, ep=nap.IntervalSet(0, off_time))

#spike count series (1 neuron)
# plot_series = nap.IntervalSet(start=song_times.start[3], end=song_times.start[3]+30)
# plt.figure(figsize=(7, 3.5))
# plt.step(count[:,0].restrict(song_times[0]).t, count[:,0].restrict(song_times[0]).d, where="post")
# plt.xlabel("Time (sec)")
# plt.ylabel("Spikes")

#choose spike history window
hist_window_sec = 0.9
hist_window_size = int(hist_window_sec * count.rate)

# SEARCH FOR BEST BASIS PARAMS
# hist_window_sec = np.arange(0.1, 3, 0.1)
# hist_window_size = [int(hist_window_sec[i] * count.rate) for i in range(len(hist_window_sec))]
# basis_fun = np.arange(2,11)

# basis = nmo.basis.RaisedCosineBasisLog(6, mode="conv", window_size=hist_window_size)
# bas = nmo.basis.TransformerBasis(basis**195)
# model = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.Ridge(regularizer_strength=0.1, solver_name="LBFGS"))
# pipe = Pipeline([("eval", bas), ("fit", model)])
#
# param_grid  = {"eval__basis":[nmo.basis.RaisedCosineBasisLog(3, mode="conv", window_size=50)**195,
#                               nmo.basis.RaisedCosineBasisLog(3, mode="conv", window_size=90)**195,
#                               nmo.basis.RaisedCosineBasisLog(6, mode="conv", window_size=50)**195,
#                               nmo.basis.RaisedCosineBasisLog(6, mode="conv", window_size=90)**195,
#                               nmo.basis.RaisedCosineBasisLog(9, mode="conv", window_size=50)**195,
#                               nmo.basis.RaisedCosineBasisLog(9, mode="conv", window_size=90)**195]}
#
# cv = GridSearchCV(pipe, param_grid)
# cv.fit(count, count.squeeze())
#
# res = cv.best_params_
# print(res['eval__basis'].n_basis_funcsn)
#
# quit()

# define  basis
basis = nmo.basis.RaisedCosineBasisLog(6, mode="conv", window_size=hist_window_size)
time, basis_kernels = basis.evaluate_on_grid(hist_window_size)

# plot basis
# plt.figure(figsize=(7, 3))
# plt.plot(basis(np.linspace(0, 1, 100)))

# define model
model = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.Ridge(regularizer_strength=0.1, solver_name="LBFGS"))

# train test split 60-40
duration = count.time_support.tot_length("s")
start = count.time_support["start"]
end = count.time_support["end"]
training = nap.IntervalSet(start, start + duration * 0.6)
testing = nap.IntervalSet(start + duration * 0.6, end)

count_train = count.restrict(training)

# create song_times_glm.py
n_bat = int(count_train.shape[0] / 10)
batches = gen_batches(count_train.shape[0], n_bat)

# train
for bat in batches:
    # apply basis to spikes
    count_bat = count_train[bat]
    X_bat = basis.compute_features(count_bat)
    print(X_bat.shape)

    model.fit(X_bat, count_bat.squeeze())


# model output
weights = model.coef_.reshape(count.shape[1], basis.n_basis_funcs, count.shape[1])
filters = np.einsum("jki,tk->ijt", weights, basis_kernels)
fr_params = model.intercept_
results_dict = {"weights": weights, "filters": filters, "fr_params": fr_params}

np.save("wr1.npy", results_dict)

# compute score
score_train = []
for bat in batches:
    # apply basis to spikes
    count_bat = count_train[bat]
    X_bat = basis.compute_features(count_bat)

    score = model.score(X_bat, count_bat.squeeze(), score_type="pseudo-r2-McFadden")
    print(score)
    score_train.append(score)

count_test = count.restrict(testing)
X_test = basis.compute_features(count_test)
score_test = model.score(X_test, count.restrict(testing).squeeze(), score_type="pseudo-r2-McFadden")

print("Score(train data):", np.mean(score_train))
print("Score(test data):", score_test)

# plot coupling weights

filter_plot = filters[:20,:20,:]

def plot_coupling(responses, cmap_name="seismic",
                      figsize=(10, 8), fontsize=15, alpha=0.5, cmap_label="hsv"):

    # plot heatmap
    sum_resp = np.sum(responses, axis=2)
    # normalize by cols (for fixed receiver neuron, scale all responses
    # so that the strongest peaks to 1)
    sum_resp_n = (sum_resp.T / sum_resp.max(axis=1)).T

    # scale to 0,1
    color = -0.5 * (sum_resp_n - sum_resp_n.min()) / sum_resp_n.min()

    cmap = plt.get_cmap(cmap_name)
    n_row, n_col, n_tp = responses.shape
    time = np.arange(n_tp)
    fig, axs = plt.subplots(n_row, n_col, figsize=figsize, sharey="row")
    for rec, rec_resp in enumerate(responses):
        for send, resp in enumerate(rec_resp):
            axs[rec, send].plot(time, responses[rec, send], color="k")
            axs[rec, send].spines["left"].set_visible(False)
            axs[rec, send].spines["bottom"].set_visible(False)
            axs[rec, send].set_xticks([])
            axs[rec, send].set_yticks([])
            axs[rec, send].axhline(0, color="k", lw=0.5)
    for rec, rec_resp in enumerate(responses):
        for send, resp in enumerate(rec_resp):
            xlim = axs[rec, send].get_xlim()
            ylim = axs[rec, send].get_ylim()
            rect = plt.Rectangle(
                (xlim[0], ylim[0]),
                xlim[1] - xlim[0],
                ylim[1] - ylim[0],
                alpha=alpha,
                color=cmap(color[rec, send]),
                zorder=1
            )
            axs[rec, send].add_patch(rect)
            axs[rec, send].set_xlim(xlim)
            axs[rec, send].set_ylim(ylim)
    axs[n_row // 2, 0].set_ylabel("receiver\n", fontsize=fontsize)
    axs[n_row-1, n_col // 2].set_xlabel("\nsender", fontsize=fontsize)

    plt.suptitle("Pairwise Interaction", fontsize=fontsize)
    return fig

plot_coupling(filter_plot)

# calculate spike predictions
spike_pred = model.predict(X_test) * count.rate

# plot predicted vs true firing rates
ep = nap.IntervalSet(testing.start, testing.start+30)
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(2,1,1)
firing_rate = spike_pred.restrict(ep).d[:,:15]
firing_rate = firing_rate.T / np.max(firing_rate, axis=1)
plt.imshow(firing_rate[::-1], cmap="Blues", aspect="auto")
ax1.set_ylabel("Neuron # (pred)")
ax2 = fig.add_subplot(2,1,2)
firing_rate = count.restrict(ep).d[:,:15]
firing_rate = firing_rate.T / np.max(firing_rate, axis=1)
plt.imshow(firing_rate[::-1], cmap="Blues", aspect="auto")
ax2.set_ylabel("Neuron # (true)")
plt.xlabel("Time (sec)")
plt.show()

