import numpy as np
import scipy.io as sio
import pynapple as nap
import nemos as nmo
import matplotlib.pyplot as plt

results_dict = np.load("/Users/macari216/Desktop/glm_songbirds/songbirds/results.npy",allow_pickle=True).item()
weights = results_dict["weights"]
filters = results_dict["filters"]
time = results_dict["time"]
spike_pred = results_dict["spike_pred"]

audio_segm = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57AudioSegments.mat')['c57AudioSegments']
song_times = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57SongTimes.mat')['c57SongTimes']
off_time = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57LightOffTime.mat')['c57LightOffTime']
spikes_all = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57SpikeTimesAll.mat')['c57SpikeTimesAll']
spikes_quiet = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']

song_times = nap.IntervalSet(start=song_times[:,0], end=song_times[:,1])
audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

ts_dict_all = {key: nap.Ts(spikes_all[key, 0].flatten()) for key in range(spikes_all.shape[0])}
spike_times_all = nap.TsGroup(ts_dict_all)

ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

audio_on = audio_segm[:149]
time_on = nap.IntervalSet(0, off_time)
time_quiet = time_on.set_diff(audio_on)
time_quiet = time_quiet.drop_short_intervals(1,'s')

binsize = 0.1
count = spike_times_quiet.count(binsize, ep=time_quiet)

model = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.Ridge(regularizer_strength=1.0, solver_name="LBFGS"))

params = ()

model.predict()

quit()

# most active neurons
quiet_rates = spike_times_quiet['rate']
quiet_thr = np.median(quiet_rates)
hf_quiet = quiet_rates[quiet_rates > quiet_thr]
hfq_id = np.array(hf_quiet.index)

# plot coupling weights for 2 neurons
plt.figure()
plt.title("Coupling Filter From Neuron 1 To Neuron 2")
plt.plot(time, filters[1,0,:], "k", lw=2)
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time from spike (sec)")
plt.ylabel("Weight")

# plot spiking rates
ep = nap.IntervalSet(8610, 8610+60)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(2,1,1)
firing_rate = spike_pred.restrict(ep).d[:,:15]
firing_rate = firing_rate.T / np.max(firing_rate, axis=1)
plt.imshow(firing_rate[::-1], cmap="Greys", aspect="auto")
ax1.set_ylabel("Neuron")
ax1.set_title("Predicted firing rate")
ax2 = fig.add_subplot(2,1,2)
firing_rate = count.restrict(ep).d[:,:15]
firing_rate = firing_rate.T / np.max(firing_rate, axis=1)
plt.imshow(firing_rate[::-1], cmap="Greys", aspect="auto")
ax2.set_ylabel("Neuron")
ax2.set_title("True firing rate")
fig.subplots_adjust(hspace=1)
plt.xlabel("Time (sec)")
fig.suptitle("Resting Data")


# plot history filters
filter_plot = filters[:20,:20,:]

def plot_coupling(responses, cmap_name="bwr",
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
plt.show()

# resp = np.matmul(X, model.coef_)
# plt.plot(resp[:,0], lw=0.5)
# plt.xlabel("time (sec)")
# plt.ylabel()

# plt.imshow(sum_resp_n[:25, :25], cmap='bwr')
# plt.colorbar()
# plt.xlabel("sender neuron")
# plt.ylabel("receiver neuron")

filters_sum = np.sum(filters, axis=2)
neg_filt = filters[filters_sum < 0]
pos_filt = filters[filters_sum > 0]

delay_neg = []
delay_pos = []

for i in range(neg_filt.shape[0]):
    delay_neg.append(np.argmax(neg_filt[i]))

for i in range(pos_filt.shape[0]):
    delay_pos.append(np.argmax(pos_filt[i]))

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4), sharex=True)
ax1.hist(delay_neg, bins=9, color='cornflowerblue', edgecolor='k')
ax1.set_xticklabels(np.round(time, 2))
ax1.set_title("negative filters")
ax1.set_ylabel("filter peak, count")
ax1.set_xlabel("time from spike (sec)")
plt.hist(delay_pos, bins=9, color='salmon', edgecolor='k')
ax2.set_title('positive filters')
ax2.set_xlabel('time from spike (sec)')

# sum_filter = np.sum(filters, axis=2)
# print(sum_filter[sum_filter<0].shape)
# print(sum_filter[sum_filter>=0].shape)
# print(sum_filter[sum_filter<0].size/sum_filter.size)

#2 halves

# duration = count.time_support.tot_length("s")
# start = count.time_support["start"]
# end = count.time_support["end"]
# first = nap.IntervalSet(start[0], start[0] + duration * 0.5)
# second = nap.IntervalSet(start[0] + duration * 0.5, end[-1])
#
# model.fit(X.restrict(first), count.restrict(first).squeeze())
# weights_fh = model.coef_.reshape(count.shape[1], basis.n_basis_funcs, count.shape[1])
# filters_fh = np.einsum("jki,tk->ijt", weights_fh, basis_kernels)
#
# model.fit(X.restrict(second), count.restrict(second).squeeze())
# weights_sh = model.coef_.reshape(count.shape[1], basis.n_basis_funcs, count.shape[1])
# filters_sh = np.einsum("jki,tk->ijt", weights_sh, basis_kernels)
#
# plt.figure()
# plt.title("Spike History Weights")
# plt.plot(time, filters_fh[0,0,:], "--k", lw=2, label="1st half")
# plt.plot(time, filters_sh[0,0,:], color="orange", lw=2, ls="--", label="2nd half")
# plt.axhline(0, color="k", lw=0.5)
# plt.xlabel("Time from spike (sec)")
# plt.ylabel("Weight")
# plt.legend()
# plt.show()
#
# diff = []
# for i in range(195):
#     for j in range(195):
#         diff.append(np.sum(filters_fh[i,j,:] - filters_sh[i,j,:]))
#
# plt.hist(diff, bins=50)
# diff[np.logical_and(diff > -0.007, diff < 0.0023)].size
#
# amplitudes_fh = np.zeros(count.shape[1])
# for i in range(count.shape[1]):
#     amplitudes_fh[i] = np.max(weights_fh[i, :, i]) - np.min(weights_fh[i, :, i])
#
# amplitudes_sh = np.zeros(count.shape[1])
# for i in range(count.shape[1]):
#     amplitudes_sh[i] = np.max(weights_sh[i, :, i]) - np.min(weights_sh[i, :, i])
#
# plt.figure()
# plt.scatter(amplitudes_fh, amplitudes_sh, color='k')
# plt.xlabel("first half")
# plt.ylabel("second half")
# plt.title("Weight Amplitudes")

# # mask for group lasso (?? maybe)
# num_groups = count.shape[1]
# num_features = count.shape[1]*hist_window_size
#
# mask = np.zeros((num_groups, num_features))
# for i in range(num_groups):
#     mask[i, i*hist_window_size:i*hist_window_size+hist_window_size] = np.ones(hist_window_size)