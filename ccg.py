import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pynapple as nap
from itertools import combinations, product
import nemos as nmo
import pandas as pd

nap.nap_config.suppress_conversion_warnings = True

results_dict = np.load("/Users/macari216/Desktop/glm_songbirds/songbirds/results_gl_sorted.npy", allow_pickle=True).item()
weights = results_dict["weights"]
filters = results_dict["filters"]
time = results_dict["time"]

audio_segm = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57LightOffTime.mat')['c57LightOffTime']
spikes_quiet = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
ei_labels = sio.loadmat('/Users/macari216/Desktop/glm_songbirds/songbirds/c57EI.mat')['c57EI']

audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

#convert times to Interval Sets and spikes to TsGroups
ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

# add E/I labels
spike_times_quiet["EI"] = ei_labels

inh = spike_times_quiet.getby_category("EI")[-1]
exc = spike_times_quiet.getby_category("EI")[1]

ee_ccg = nap.compute_crosscorrelogram(exc, 0.005, 0.03, ep=nap.IntervalSet(0, off_time))
ei_ccg = nap.compute_crosscorrelogram((exc,inh), 0.005, 0.03, ep=nap.IntervalSet(0, off_time))
ie_ccg = nap.compute_crosscorrelogram((inh,exc), 0.005, 0.03, ep=nap.IntervalSet(0, off_time))
ii_ccg = nap.compute_crosscorrelogram(inh, 0.005, 0.03, ep=nap.IntervalSet(0, off_time))

# ee_ccg_np = (ee_ccg.to_numpy().reshape((11,101,101))).transpose((1,2,0))
# ei_ccg_np = (ei_ccg.to_numpy().reshape((11,101,94))).transpose((1,2,0))
# ie_ccg_np = (ie_ccg.to_numpy().reshape((11,94,101))).transpose((1,2,0))
# ii_ccg_np = (ii_ccg.to_numpy().reshape((11,94,94))).transpose((1,2,0))

filters = filters.transpose((1,0,2))

filters_ee = filters[:101, :101]
filters_ei = filters[:101, -94:]
filters_ie = filters[-94:, :101]
filters_ii = filters[-94:, -94:]

ee_max_idx = ee_ccg.max().sort_values(ascending=False).index

ei_max_ampl = (ei_ccg.max() - ei_ccg.min()).sort_values(ascending=False).index

plt.figure()
for i in range(10):
    plt.plot(np.linspace(0,0.01,11), ei_ccg[ei_max_ampl[i]].iloc[-11:], label=ei_max_ampl[i])
    plt.ylabel("post-synaptic FR (I)")
    plt.xlabel("time after pre-synaptic spike (E)")
    plt.title("CCG E/I")
    plt.legend()

plt.figure()
for i in range(10):
    plt.plot(np.linspace(0,0.025,6), ei_ccg[ei_max_ampl[-(i+1)]].iloc[-6:], label=ei_max_ampl[-(i+1)])
    plt.ylabel("post-synaptic FR (I)")
    plt.xlabel("time after pre-synaptic spike (E)")
    plt.title("CCG E/I")
    plt.legend()

plt.figure()
for idx in ei_max_ampl[:10]:
    send = idx[0]
    rec = idx[1]
    filter = filters[send, rec, :]
    plt.plot(time, filter, label=idx)
    plt.ylabel("weight (I)")
    plt.xlabel("time after pre-synaptic spike (E)")
    plt.title("E/I filters")
    plt.legend()

plt.figure()
for idx in ei_max_ampl[-10:]:
    send = idx[0]
    rec = idx[1]
    filter = filters[send, rec, :]
    plt.plot(time, filter, label=idx)
    plt.ylabel("weight (I)")
    plt.xlabel("time after pre-synaptic spike (E)")
    plt.title("E/I filters")
    plt.legend()

plt.show()

zero_idx = np.where(filters.sum(axis=2) == 0)

zero_ccg = []
for i,j in zip(zero_idx[0], zero_idx[1]):
    if (i,j) in ei_ccg.keys():
        ccg_i = ei_ccg[(i,j)]
        ampl = ccg_i.max() - ccg_i.min()
        zero_ccg.append(ampl)

zero_ccg = np.array(zero_ccg)
plt.hist(zero_ccg, bins=20)

ei_ampl = np.array(ie_ccg.max()-ie_ccg.min())

print(zero_ccg[zero_ccg<0.1].size / zero_ccg.size)
print(ei_ampl[ei_ampl<0.1].size / ei_ampl.size)

filt_ei_ampl = np.where(np.sort(filters_ei.max(axis=2) - filters_ei.min(axis=2)))


# print(zero_ccg[zero_ccg<0.1].size / zero_ccg.size)
# print(ei_ampl[ei_ampl<0.1].size / ei_ampl.size)
# 0.07946069994262765
# 0.06562039182641669

pairs = list(combinations(exc.keys(), 2))
ee_ccg_counts = nap.compute_crosscorrelogram(exc, 0.0001, 0.01, ep=nap.IntervalSet(0, off_time), norm=False)

C = {}
for i,j in pairs:
    t1 = exc.restrict(nap.IntervalSet(0, off_time))[i].index
    nt1 = len(t1)
    C[(i,j)] = ee_ccg_counts[(i,j)] * (nt1*0.0001)
C = pd.DataFrame.from_dict(C)


n1 = 28
n2 = 179
plt.figure()
plt.bar(C[(n1,n2)].index, C[(n1,n2)], width=0.0001)
plt.vlines(0, 0, C[(n1,n2)].max(), 'r', '--')
plt.xlabel("lag (ms)")
plt.ylabel("spike count")
plt.title(f"CCG between {n1} (ref) and {n2}  (target)")