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

#
# # plot history filters
# filter_plot = filters[:20,:20,:]
#
# def plot_coupling(responses, cmap_name="seismic",
#                       figsize=(8, 6), fontsize=15, alpha=0.5, cmap_label="hsv"):
#
#     # plot heatmap
#     sum_resp = np.sum(responses, axis=2)
#     # normalize by cols (for fixed receiver neuron, scale all responses
#     # so that the strongest peaks to 1)
#     sum_resp_n = (sum_resp.T / sum_resp.max(axis=1)).T
#
#     # scale to 0,1
#     color = -0.5 * (sum_resp_n - sum_resp_n.min()) / sum_resp_n.min()
#
#     cmap = plt.get_cmap(cmap_name)
#     n_row, n_col, n_tp = responses.shape
#     time = np.arange(n_tp)
#     fig, axs = plt.subplots(n_row, n_col, figsize=figsize, sharey="row")
#     for rec, rec_resp in enumerate(responses):
#         for send, resp in enumerate(rec_resp):
#             axs[rec, send].plot(time, responses[rec, send], color="k")
#             axs[rec, send].spines["left"].set_visible(False)
#             axs[rec, send].spines["bottom"].set_visible(False)
#             axs[rec, send].set_xticks([])
#             axs[rec, send].set_yticks([])
#             axs[rec, send].axhline(0, color="k", lw=0.5)
#     for rec, rec_resp in enumerate(responses):
#         for send, resp in enumerate(rec_resp):
#             xlim = axs[rec, send].get_xlim()
#             ylim = axs[rec, send].get_ylim()
#             rect = plt.Rectangle(
#                 (xlim[0], ylim[0]),
#                 xlim[1] - xlim[0],
#                 ylim[1] - ylim[0],
#                 alpha=alpha,
#                 color=cmap(color[rec, send]),
#                 zorder=1
#             )
#             axs[rec, send].add_patch(rect)
#             axs[rec, send].set_xlim(xlim)
#             axs[rec, send].set_ylim(ylim)
#     axs[n_row // 2, 0].set_ylabel("receiver\n", fontsize=fontsize)
#     axs[n_row-1, n_col // 2].set_xlabel("\nsender", fontsize=fontsize)
#
#     plt.suptitle("Pairwise Interaction", fontsize=fontsize)
#     return fig
#
# plot_coupling(filter_plot)
#
#
# # most active neurons
# quiet_rates = spike_times_quiet['rate']
# quiet_thr = np.median(quiet_rates)
# hf_quiet = quiet_rates[quiet_rates > quiet_thr]
# hfq_id = np.array(hf_quiet.index)
#
# # plot spiking rates
# spike_pred = model.predict(X.restrict(testing))
# ep = nap.IntervalSet(testing.start, testing.start+30)
#
# fig = plt.figure(figsize=(8,4))
# ax1 = fig.add_subplot(2,1,1)
# firing_rate = spike_pred.restrict(ep).d[:,hfq_id[:15]]
# firing_rate = firing_rate.T / np.max(firing_rate, axis=1)
# plt.imshow(firing_rate[::-1], cmap="Greys", aspect="auto")
# ax1.set_ylabel("Neuron")
# ax1.set_title("Predicted firing rate")
# ax2 = fig.add_subplot(2,1,2)
# firing_rate = count.restrict(ep).d[:,hfq_id[:15]]
# firing_rate = firing_rate.T / np.max(firing_rate, axis=1)
# plt.imshow(firing_rate[::-1], cmap="Greys", aspect="auto")
# ax2.set_ylabel("Neuron")
# ax2.set_title("True firing rate")
# fig.subplots_adjust(hspace=1)
# plt.xlabel("Time (sec)")
# plt.show()
#

