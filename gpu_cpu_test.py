import numpy as np
import scipy.io as sio
import pynapple as nap
import nemos as nmo
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--Neuron", help="Specify GLM input neuron (0-194)")
args = parser.parse_args()

nap.nap_config.suppress_conversion_warnings = True

audio_segm = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57LightOffTime.mat')['c57LightOffTime']
spikes_quiet = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
ei_labels = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57EI.mat')['c57EI']

audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

spike_times_quiet["EI"] = ei_labelstime_on = nap.IntervalSet(0, off_time).set_diff(audio_segm)

training_end = off_time * 0.8
time_quiet_train = nap.IntervalSet(0, training_end).set_diff(audio_segm)
time_quiet_test = nap.IntervalSet(training_end, off_time).set_diff(audio_segm)

# output lists
score_train = np.zeros(1)
score_test = np.zeros(1)

binsize = 0.0001

print(f"before count_train: {datetime.now().time()}")
count_train = spike_times_quiet.count(binsize, ep=time_quiet_train)
print(f"count_train: {datetime.now().time()}")

hist_window_sec = 0.004
hist_window_size = int(hist_window_sec * count_train.rate)

n_fun = 9
basis = nmo.basis.RaisedCosineBasisLog(n_fun, mode="conv", window_size=hist_window_size)
time, basis_kernels = basis.evaluate_on_grid(hist_window_size)
time *= hist_window_sec

n_bat = 1000
batch_size = time_quiet_train.tot_length() / n_bat

def batcher(start):
    end = start + batch_size
    ep = nap.IntervalSet(start, end)
    start = end
    print(f"before restrict X: {datetime.now().time()}")
    X_counts = count_train.restrict(ep)
    print(f"before convolution: {datetime.now().time()}")
    X = basis.compute_features(X_counts)
    print(f"before restrict Y: {datetime.now().time()}")
    Y_counts = count_train[:, int(args.Neuron)].restrict(ep)
    return X, Y_counts.squeeze(), start

model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized(
    solver_name="GradientDescent", solver_kwargs={"stepsize": 0.2, "acceleration": False}))
start = time_quiet_train.start[0]
params, state = model.initialize_solver(*batcher(start))

for i in range(10):
    # Get a batch of data
    X, Y, start = batcher(start)
    print(f"received X and Y: {datetime.now().time()}")

    # Do one step of gradient descent.
    params, state = model.update(params, state, X, Y)
    print(f"one model step: {datetime.now().time()}")
    print(f"{i}------------------------")

