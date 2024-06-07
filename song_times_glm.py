import numpy as np
import scipy.io as sio
import pynapple as nap
import nemos as nmo
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

audio_segm = sio.loadmat('/Users/macari216/Desktop/glm-songbirds/songbirds/c57AudioSegments.mat')['c57AudioSegments']
song_times = sio.loadmat('/Users/macari216/Desktop/glm-songbirds/songbirds/c57SongTimes.mat')['c57SongTimes']
off_time = sio.loadmat('/Users/macari216/Desktop/glm-songbirds/songbirds/c57LightOffTime.mat')['c57LightOffTime']
spikes_all = sio.loadmat('/Users/macari216/Desktop/glm-songbirds/songbirds/c57SpikeTimesAll.mat')['c57SpikeTimesAll']
spikes_quiet = sio.loadmat('/Users/macari216/Desktop/glm-songbirds/songbirds/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']

song_times = nap.IntervalSet(start=song_times[:,0], end=song_times[:,1])
audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

ts_dict_all = {key: nap.Ts(spikes_all[key, 0].flatten()) for key in range(spikes_all.shape[0])}
spike_times_all = nap.TsGroup(ts_dict_all)


binsize = 0.005   # in seconds
# count = spike_times_quiet[spike_times_quiet.subset==0].count(binsize, ep=nap.IntervalSet(0, off_time))
count = spike_times_all.count(binsize, ep=song_times)

hist_window_sec = 0.05
hist_window_size = int(hist_window_sec * count.rate)
print(count.rate)

basis = nmo.basis.RaisedCosineBasisLog(9, mode="conv", window_size=hist_window_size)
time, basis_kernels = basis.evaluate_on_grid(hist_window_size)

X = basis.compute_features(count)
print(X.shape)

duration = X.time_support.tot_length("s")
start = X.time_support["start"]
end = X.time_support["end"]
training =nap.IntervalSet(song_times.start[0], song_times.end[21])
testing =nap.IntervalSet(song_times.start[21], song_times.end[29])

model = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.Ridge(regularizer_strength=1., solver_name="LBFGS"))
model.fit(X.restrict(training), count.restrict(training).squeeze())

score_train = model.score(X.restrict(training), count.restrict(training).squeeze(), score_type="pseudo-r2-McFadden")
score_test = model.score(X.restrict(testing), count.restrict(testing).squeeze(), score_type="pseudo-r2-McFadden")
print("Score(train data):", score_train)
print("Score(test data):", score_test)


weights = model.coef_.reshape(count.shape[1], basis.n_basis_funcs, count.shape[1])
filters = np.einsum("jki,tk->ijt", weights, basis_kernels)

# plot history filters
filter_plot = filters[:10,:10,:]

def plot_coupling(responses, cmap_name="seismic",
                      figsize=(8, 6), fontsize=15, alpha=0.5, cmap_label="hsv"):

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


# most active neurons
song_rates = spike_times_all.restrict(song_times)['rate']
song_thr = np.median(song_rates)
hf_song = song_rates[song_rates > song_thr]
hfs_id = np.array(hf_song.index)

# plot spiking rates
spike_pred = model.predict(X.restrict(testing)) * count.rate
ep = nap.IntervalSet(testing.start, testing.start+30)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(2,1,1)
firing_rate = spike_pred.restrict(ep).d[:,hfs_id[:15]]
firing_rate = firing_rate.T / np.max(firing_rate, axis=1)
plt.imshow(firing_rate[::-1], cmap="Blues", aspect="auto")
ax1.set_ylabel("Neuron")
ax1.set_title("Predicted firing rate")
ax2 = fig.add_subplot(2,1,2)
firing_rate = count.restrict(ep).d[:,hfs_id[:15]]
firing_rate = firing_rate.T / np.max(firing_rate, axis=1)
plt.imshow(firing_rate[::-1], cmap="Blues", aspect="auto")
ax2.set_ylabel("Neuron")
ax2.set_title("True firing rate")
plt.xlabel("Time (sec)")
plt.show()
