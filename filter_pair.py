import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pynapple as nap
import nemos as nmo
from itertools import combinations, product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n1", "--Neuron1", help="post-synaptic neuron")
parser.add_argument("-n2", "--Neuron2", help="pre-synaptic neuron")
args = parser.parse_args()


rows = [int(args.Neuron1),int(args.Neuron2)]

nap.nap_config.suppress_conversion_warnings = True

audio_segm = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57LightOffTime.mat')['c57LightOffTime']
spikes_quiet = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
ei_labels = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57EI.mat')['c57EI']

audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])
ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in rows}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

subset_labels = []
for idx in range(195):
    if idx in rows:
        subset_labels.append(ei_labels[idx])

spike_times_quiet["EI"] = subset_labels

training_end = off_time * 0.7
time_quiet_train = nap.IntervalSet(0, training_end).set_diff(audio_segm)
time_quiet_test = nap.IntervalSet(training_end, off_time).set_diff(audio_segm)

binsize = 0.0001
count_train = spike_times_quiet.count(binsize, ep=time_quiet_train)

n_neurons = count_train.shape[1]

hist_window_sec = 0.004
hist_window_size = int(hist_window_sec * count_train.rate)

n_fun =9

basis = nmo.basis.RaisedCosineBasisLog(n_fun, mode="conv", window_size=hist_window_size)
time, basis_kernels = basis.evaluate_on_grid(hist_window_size)
time *= hist_window_sec

n_bat = 1000
batch_size = time_quiet_train.tot_length() / n_bat

def batcher(start):
    end = start + batch_size
    ep = nap.IntervalSet(start, end)
    start = end
    counts = count_train.restrict(ep)
    X = basis.compute_features(counts)
    return X, counts, start

model = nmo.glm.PopulationGLM(regularizer=nmo.regularizer.UnRegularized(
    solver_name="GradientDescent", solver_kwargs={"stepsize": 0.2, "acceleration": False}))
start = time_quiet_train.start[0]
params, state = model.initialize_solver(*batcher(start))

n_ep = 25

score_train = np.zeros(n_ep)

for ep in range(n_ep):
    start = time_quiet_train.start[0]
    for i in range(n_bat):
        # Get a batch of data
        X, Y, start = batcher(start)
        # Do one step of gradient descent.
        params, state = model.update(params, state, X, Y)
    # Score the model along the time axis
    score_train[ep] = model.score(X, Y, score_type="log-likelihood")
    if ep%5 ==0:
        print(f"Epoch: {ep}, train ll:{score_train[ep]}")


weights = model.coef_.reshape(n_neurons, basis.n_basis_funcs, n_neurons)
filters = np.einsum("jki,tk->ijt", weights, basis_kernels)

ccg = nap.compute_crosscorrelogram((spike_times_quiet,spike_times_quiet), 0.0001, 0.004, norm=False)
pairs = list(combinations(spike_times_quiet.keys(), 2))

ccg_counts = {}
for i,j in pairs:
    t1 = spike_times_quiet[i].index
    nt1 = len(t1)
    ccg_counts[(i,j)] = ccg[(i,j)] * (nt1*0.0001)
ccg_counts = pd.DataFrame.from_dict(ccg_counts)

def ccg_filt_plot(counts):
    n1 = rows[0]
    n2 = rows[1]
    x_ticks = np.round(counts[(n1, n2)].index, 3)
    filter = filters[1, 0, :]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(counts[(n1, n2)].index, counts[(n1, n2)], width=0.0001)
    ax1.axvline(0, 0, counts[(n1, n2)].max(), color='r', ls='--')
    ax1.set_xticks(x_ticks)
    ax1.set_xlabel("lag (ms)")
    ax1.set_ylabel("spike count")
    ax1.set_title(f"CCG between {n1} (ref) and {n2} (target)")
    ax2.plot(time, filter, label=(n1,n2))
    ax2.set_xlabel("lag (ms)")
    ax2.set_ylabel("gain")
    ax2.set_xticks(x_ticks[x_ticks>=0])
    ax2.set_title(f"Filter from {n1} to {n2}")
    fig.subplots_adjust(wspace=0.3)
    return fig


plt.figure()
plt.plot(score_train)
plt.xlabel("Epoch")
plt.ylabel("log-likelihood")

plt.show()
