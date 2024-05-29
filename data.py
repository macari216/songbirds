import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt

song_times = sio.loadmat('c57SongTimes.mat')['c57SongTimes']
audio_segm = sio.loadmat('c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('c57LightOffTime.mat')['c57LightOffTime']

spikes_all = sio.loadmat('c57SpikeTimesAll.mat')['c57SpikeTimesAll']
spikes_quiet = sio.loadmat('c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']


