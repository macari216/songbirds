import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt

song_times = sio.loadmat('c57SongTimes.mat')['c57SongTimes']
audio_segm = sio.loadmat('c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('c57LightOffTime.mat')['c57LightOffTime']

spikes_all = sio.loadmat('c57SpikeTimesAll.mat')['c57SpikeTimesAll']
spikes_quiet = sio.loadmat('c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']


# idx = 100
#
# arr = np.array([[1],[2],[3]])
# spikeslist = spikes_all.tolist()
# spike_times = spikeslist[idx][0]
# print(np.shape(spike_times))
#
# song = np.arange(song_times[1,0], song_times[1,1], 1e-8)
# spike_times_plot = spike_times[(spike_times>=song[0]) & (spike_times<song[-1])]
#
# fig1, ax1 = plt.subplots(1)
# ax1.plot(spike_times_plot, [1]*spike_times_plot.size, 'ko')
# plt.show()
