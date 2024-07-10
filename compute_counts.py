import scipy.io as sio
import pynapple as nap
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--Neuron", help="Specify GLM input neuron (0-194)")
args = parser.parse_args()

nap.nap_config.set_backend("jax")

print(f"before loading: {datetime.now().time()}")
audio_segm = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57LightOffTime.mat')['c57LightOffTime']
spikes_quiet = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
print(f"loaded: {datetime.now().time()}")

audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])
ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

time_on = nap.IntervalSet(0, off_time).set_diff(audio_segm)

one_int = off_time*0.8
train_int = nap.IntervalSet(0, one_int).set_diff(audio_segm)

n_bat = 1000
batch_size = train_int.tot_length() / n_bat

binsize=0.0001

print("COMPUTING BATCHED COUNTS")
start = 0.0
def batcher(start):
    end = start + batch_size
    ep = nap.IntervalSet(start, end)
    start = end
    print(f"before computing batches counts: {datetime.now().time()}")
    X_counts = spike_times_quiet.count(binsize, ep=ep)
    print(f"after computing batched X counts: {datetime.now().time()}")
    Y_counts = spike_times_quiet[int(args.Neuron)].count(binsize, ep=ep)
    print(f"after computing batched Y counts: {datetime.now().time()}")
    return X_counts, Y_counts, start


for i in range(10):
    X, Y, start = batcher(start)
    print(f"after returning X and Y: {datetime.now().time()}")

print("COMPUTING ALL COUNTS")
print(f"before computing counts: {datetime.now().time()}")
counts = spike_times_quiet.count(binsize, ep=time_on)
print(f"after computing counts: {datetime.now().time()}")

c