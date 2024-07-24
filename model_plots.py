import numpy as np
import scipy.io as sio
import pynapple as nap
import nemos as nmo
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations, product
import seaborn as sns

results_dict = np.load("/songbirds/results.npy", allow_pickle=True).item()
weights = results_dict["weights"]
filters = results_dict["filters"]
time = results_dict["time"]
spike_pred = results_dict["spike_pred"]

song_times = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57SongTimes.mat')['c57SongTimes']
audio_segm = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57LightOffTime.mat')['c57LightOffTime']
spikes_quiet = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
spikes_all = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57SpikeTimesAll.mat')['c57SpikeTimesAll']

song_times = nap.IntervalSet(start=song_times[:,0], end=song_times[:,1])
audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

audio_on = audio_segm[:149]
training_end = off_time * 0.7
time_on_train = nap.IntervalSet(0, training_end)
time_on_test = nap.IntervalSet(training_end, off_time)
time_quiet_train = time_on_train.set_diff(audio_on)
time_quiet_train = time_quiet_train.drop_short_intervals(1,'s')
time_quiet_test = time_on_test.set_diff(audio_on)
time_quiet_test = time_quiet_test.drop_short_intervals(1,'s')

ts_dict_all = {key: nap.Ts(spikes_all[key, 0].flatten()) for key in range(spikes_all.shape[0])}
spike_times_all = nap.TsGroup(ts_dict_all)

ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

binsize = 0.005
count_test = spike_times_quiet.count(binsize, ep=time_quiet_test)

# most active neurons
quiet_rates = spike_times_quiet['rate']
quiet_thr = np.median(quiet_rates)
hf_quiet = quiet_rates[quiet_rates > quiet_thr]
hfq_id = np.array(hf_quiet.index)

# plot coupling weights for 2 neurons
n1 = 91
n2 = 90
plt.figure()
plt.title(f"Coupling Filter From Neuron {n1} To Neuron {n2+1}")
plt.title(f"Coupling Filter From Neuron {n1} To Neuron {n2+1}")
plt.plot(time, filters[n1,n2,:], "k", lw=2)
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time from spike (sec)")
plt.ylabel("Weight")

# plot spiking rates
ep = nap.IntervalSet(8716, 8716+60)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(2,1,1)
firing_rate = spike_pred.restrict(ep).d[:,:15]
firing_rate = firing_rate.T / np.max(firing_rate, axis=1)
plt.imshow(firing_rate[::-1], cmap="Greys", aspect="auto")
ax1.set_ylabel("Neuron")
ax1.set_title("Predicted firing rate")
ax2 = fig.add_subplot(2,1,2)
firing_rate = count_test.restrict(ep).d[:,:15]
firing_rate = firing_rate.T / np.max(firing_rate, axis=1)
plt.imshow(firing_rate[:,:-1], cmap="Greys", aspect="auto")
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

sum_filt = np.sum(filters, axis=2)
sum_filt_n = (sum_filt.T / np.abs(sum_filt).max(axis=1)).T
plt.figure()
plt.imshow(sum_filt_n, cmap='bwr')
plt.colorbar()
plt.xlabel("pre-synaptic")
plt.ylabel("post-synaptic")
plt.title("inferred filters (group lasso)")

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

# n=3
# ind = np.arange(n)
# width = 0.27
#
# plt.figure()
# for i in range(5):
#     plt.bar(ind+width*i, fractions[i], width)


# filter error bars
# peak_ee = np.zeros((101,101))
# for i in range(101):
#     for j in range(101):
#         ij_peak = np.argmax(np.abs(filters_ee), axis=2)[i,j]
#         peak_ee[i,j] = filters_ee[i,j,ij_peak]

# mean_ee_exc = filters_ee[peak_ee>0].mean(axis=0)
# lower_ee_exc = mean_ee_exc - filters_ee[peak_ee>0].std(axis=0)
# upper_ee_exc = mean_ee_exc + filters_ee[peak_ee>0].std(axis=0)

# plt.figure()
# plt.plot(time, filters_ee[peak_ee>0].mean(axis=0))
# mean_ee_exc = filters_ee[peak_ee>0].mean(axis=0)
# lower_ee_exc = mean_ee_exc - filters_ee[peak_ee>0].std(axis=0)
# upper_ee_exc = mean_ee_exc + filters_ee[peak_ee>0].std(axis=0)
# fig, ax = plt.subplots(figsize=(9,5))
# ax.plot(time, mean_ee_exc, label='filter mean')
# ax.plot(time, lower_ee_exc, color='tab:blue', alpha=0.1)
# ax.plot(time, upper_ee_exc, color='tab:blue', alpha=0.1)
# ax.fill_between(time, lower_ee_exc, upper_ee_exc, alpha=0.2)
# ax.set_xlabel('time from spike')
# ax.set_ylabel('filter')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.show()

filters_exc = filters[:,:101,:]
filters_inh = filters[:,-94:,:]

n_conx_exc = [ (filters_exc[:,i,:])[(filters_exc[:,i,:]).sum(axis=1)!=0].shape[0] for i in range(101)]
n_conx_exc = np.array(n_conx_exc)
n_conx_inh = [ (filters_inh[:,i,:])[(filters_inh[:,i,:]).sum(axis=1)!=0].shape[0] for i in range(94)]
n_conx_inh = np.array(n_conx_inh)

color_exc = np.where(n_conx_exc >= np.quantile(n_conx_exc, 0.9), 'r', 'mistyrose')
color_inh = np.where(n_conx_inh >= np.quantile(n_conx_inh, 0.9), 'b', 'lightsteelblue')

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
ax1.bar(np.arange(1,102), n_conx_exc, color=color_exc)
ax1.set_xlabel("pre-synaptic neuron")
ax1.set_ylabel("number of connections")
ax1.set_title("putative excitatory neurons")
ax2.bar(np.arange(1,95), n_conx_inh, color=color_inh)
ax2.set_xlabel("pre-synaptic neuron")
ax2.set_title("putative inhibitory neurons")
fig.suptitle("candidate hub neurons")

filters = np.concatenate((filters, np.zeros((40,1))), axis=1)
filter_plot = (filters.reshape(40,14,14)).transpose(1,2,0)

def comp_ccg(n1, n2):
    n1_spikes = {n1: nap.Ts(spikes_quiet[n1, 0].flatten())}
    n1_spikes = nap.TsGroup(n1_spikes)
    n2_spikes = {n2: nap.Ts(spikes_quiet[n2, 0].flatten())}
    n2_spikes = nap.TsGroup(n2_spikes)
    ccg = nap.compute_crosscorrelogram((n1_spikes, n2_spikes), 0.0001, 0.004, norm=False)
    ccg_counts = {}
    t1 = n1_spikes[n1].index
    nt1 = len(t1)
    ccg_counts[(n1, n2)] = ccg[(n1, n2)] * (nt1 * 0.0001)
    ccg_counts = pd.DataFrame.from_dict(ccg_counts)
    return ccg_counts


def ccg_filt_plot(n1, n2, counts, filters):
    x_ticks = np.round(counts[(n1, n2)].index, 3)
    filter = filters[n2,n1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(counts[(n1, n2)].index, counts[(n1, n2)], width=0.0001)
    ax1.axvline(0, 0, 1, color='r', ls='--')
    ax1.set_xticks(x_ticks, labels=(x_ticks*1000).astype(int))
    ax1.set_xlabel("lag (ms)")
    ax1.set_ylabel("spike count")
    ax1.set_title(f"CCG between {n1} (ref) and {n2} (target)")
    ax2.plot(time, filter, label=(n1,n2))
    ax2.set_xlabel("lag (ms)")
    ax2.set_ylabel("gain")
    x_ticks = x_ticks[x_ticks>=0]
    ax2.set_xticks(x_ticks, labels=(x_ticks*1000).astype(int))
    ax2.set_title(f"Filter from {n1} to {n2}")
    fig.subplots_adjust(wspace=0.3)
    return fig

# ACG
def comp_acg(n1):
    n1_spikes = {n1: nap.Ts(spikes_quiet[n1, 0].flatten())}
    n1_spikes = nap.TsGroup(n1_spikes)
    acg = nap.compute_autocorrelogram(n1_spikes, 0.0001, 0.15, norm=False)
    acg_counts = {}
    t1 = n1_spikes[n1].index
    nt1 = len(t1)
    acg_counts[n1] = acg * (nt1 * 0.0001)
    acg_counts = pd.DataFrame.from_dict(ccg_counts)
    return acg_counts

def acg_filt_plot(n1, counts, filters):
    x_ticks = np.array([-0.15,0.15])
    filter = filters[:,n1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(counts[n1].index, counts[n1], width=0.0001)
    ax1.axvline(0, 0, 1, color='r', ls='--')
    ax1.set_xticks(x_ticks, labels=(x_ticks*1000).astype(int))
    ax1.set_xlabel("lag (ms)")
    ax1.set_ylabel("spike count")
    ax1.set_title(f"ACG for Neuron {n1}")
    ax2.plot(time, filter, label=(n1))
    ax2.set_xlabel("lag (ms)")
    ax2.set_ylabel("gain")
    x_ticks = np.arange(0,0.005,0.001)
    ax2.set_xticks(x_ticks, labels=(x_ticks*1000).astype(int))
    ax2.set_title(f"Self-to-self filter, Neuron {n1}")
    fig.subplots_adjust(wspace=0.3)
    return fig


ccg = nap.compute_crosscorrelogram((spike_times_quiet,spike_times_quiet), 0.0001, 0.004, norm=False)
acg = nap.compute_autocorrelogram(spike_times_quiet, 0.0001, 0.15, norm=False)

pairs = product(list(spike_times_quiet),list(spike_times_quiet))
ccg_counts = {}
for i,j in pairs:
    t1 = spike_times_quiet[i].index
    nt1 = len(t1)
    ccg_counts[(i,j)] = ccg[(i,j)] * (nt1*0.0001)
ccg_counts = pd.DataFrame.from_dict(ccg_counts)

for i in range(195):
    ccg=ccg.drop(labels=(i,i),axis=1)

frates_sorted = ccg.max(0).sort_values(ascending=False)

acg_counts ={}
for i in range(195):
    t1 = spike_times_quiet[i].index
    nt1 = len(t1)
    acg_counts[i] = acg[i] * (nt1*0.0001)
acg_counts = pd.DataFrame.from_dict(acg_counts)

ccg_counts_sum = ccg_counts.sum(0)
counts_sorted = ccg_counts_sum.sort_values(ascending=False)

plt.figure()
plt.bar(ccg_counts[(124, 125)].index, ccg_counts[(124, 125)], width=0.0001)
plt.vlines(0, 0, max(ccg_counts[(124, 125)]), color='r', ls='--')
sns.kdeplot(x=ccg_counts[(124,125)].index, weights=ccg_counts[(124,125)].values, color='yellow', bw_adjust=0.03, cut=0)

