import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pynapple as nap
import nemos as nmo

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

# spike count
binsize = 0.005   # in seconds

# spike count series (1 neuron)
count = spike_times_all.count(binsize, ep=song_times)

plot_series = nap.IntervalSet(start=song_times.start[3], end=song_times.start[3]+30)
plt.figure(figsize=(7, 3.5))
plt.step(count[:,0].restrict(song_times[0]).t, count[:,0].restrict(song_times[0]).d, where="post")
plt.xlabel("Time (sec)")
plt.ylabel("Spikes")


# firing rates
quiet_rates = spike_times_quiet['rate']
song_rates = spike_times_all.restrict(song_times)['rate']

# firing rate distribution
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,4))
ax1.hist(quiet_rates, bins=40, edgecolor='k')
ax1.set_title('Firing Rates (quiet)')
ax1.set_ylabel('Number of Neurons')
ax1.set_xlabel('Firing Rate (Hz)')
ax2.hist(song_rates, bins=40, edgecolor='k')
ax2.set_title('Firing Rates (song)')
ax2.set_xlabel('Firing Rate (Hz)')


song_thr = np.median(song_rates)
quiet_thr = np.median(quiet_rates)

hf_song = song_rates[song_rates > song_thr]
hf_quiet = quiet_rates[quiet_rates > quiet_thr]
hfs_id = np.array(hf_song.index)
hfq_id = np.array(hf_quiet.index)
match_id = np.intersect1d(hfs_id, hfq_id)

print(song_rates.loc[match_id])
print(quiet_rates.loc[match_id])

# plot basis
#choose spike history window
hist_window_sec = 0.5
hist_window_size = int(hist_window_sec * count.rate)

# define  basis
basis = nmo.basis.RaisedCosineBasisLog(6, mode="conv", window_size=hist_window_size)
time, basis_kernels = basis.evaluate_on_grid(hist_window_size)

# plot basis
plt.figure(figsize=(7, 3))
plt.plot(basis(np.linspace(0, 1, 100)))

plt.show()

