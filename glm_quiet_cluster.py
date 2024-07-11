from datetime import datetime
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pynapple as nap
import nemos as nmo
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--Neuron", help="Specify GLM input neuron (0-194)")
args = parser.parse_args()

nap.nap_config.suppress_conversion_warnings = True

audio_segm = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57AudioSegments.mat')['c57AudioSegments']
off_time = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57LightOffTime.mat')['c57LightOffTime']
spikes_quiet = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57SpikeTimesQuiet.mat')['c57SpikeTimesQuiet']
ei_labels = sio.loadmat('/mnt/home/amedvedeva/ceph/songbird_data/c57EI.mat')['c57EI']

#convert times to Interval Sets and spikes to TsGroups
audio_segm = nap.IntervalSet(start=audio_segm[:,0], end=audio_segm[:,1])

# ts_dict_quiet = {int(args.Neuron): nap.Ts(spikes_quiet[int(args.Neuron), 0].flatten())}
ts_dict_quiet = {key: nap.Ts(spikes_quiet[key, 0].flatten()) for key in range(spikes_quiet.shape[0])}
spike_times_quiet = nap.TsGroup(ts_dict_quiet)

# add E/I label
spike_times_quiet["EI"] = ei_labels

# time intervals for CV
kf = 1

time_on = nap.IntervalSet(0, off_time).set_diff(audio_segm)
tests = []
t_st = 0
one_int = off_time*0.2
for i in range(kf):
    t_end = t_st + one_int
    test = nap.IntervalSet(t_st, t_end).set_diff(audio_segm)
    tests.append(test)
    t_st = t_end

# choose spike history window
hist_window_sec = 0.004
mean_rate = 9999
hist_window_size = int(hist_window_sec * mean_rate)

# define  basis
n_fun = 9
basis = nmo.basis.RaisedCosineBasisLog(n_fun, mode="conv", window_size=hist_window_size)
time, basis_kernels = basis.evaluate_on_grid(hist_window_size)
time *= hist_window_sec

#training epochs
n_ep = 30

# output lists
score_train = np.zeros((kf, n_ep))
score_test = np.zeros((kf, n_ep))
weights = []
filters = []
intercepts = []

# compute train and test counts
binsize = 0.0001

print(f"before epoch 0: {datetime.now().time()}")
for k, test_int in enumerate(tests):
    #count_test = spike_times_quiet.count(binsize, ep=test_int)

    train_int = time_on.set_diff(test_int)

    #X_test = basis.compute_features(count_test)

    # implement minibatching
    n_bat = 1000
    batch_size = train_int.tot_length() / n_bat

    def batcher(start):
        end = start + batch_size
        ep = nap.IntervalSet(start, end)

        while not train_int.intersect(ep):
            start += batch_size
            end += batch_size
            ep = nap.IntervalSet(start, end)
            if end > train_int.end[-1]:
                break

        start = end
        #print(f"before computing batches counts: {datetime.now().time()}")
        X_counts = spike_times_quiet.count(binsize, ep=ep)
        #print(f"after computing batched X counts: {datetime.now().time()}")
        Y_counts = spike_times_quiet[int(args.Neuron)].count(binsize, ep=ep)
        #print(f"after computing batched Y counts: {datetime.now().time()}")
        X = basis.compute_features(X_counts)
        #print(f"after convolution: {datetime.now().time()}")
        return X, Y_counts.squeeze(), start

    # define and initialize model
    model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized(
        solver_name="GradientDescent", solver_kwargs={"stepsize": 0.2, "acceleration": False}))
    start = train_int.start[0]
    #print(f"before model init: {datetime.now().time()}")
    params, state = model.initialize_solver(*batcher(start))
    #print(f"after model init: {datetime.now().time()}")

    # train model
    for ep in range(n_ep):
        start = train_int.start[0]
        # for i in range(n_bat):
        for i in range(n_bat):
            # Get a batch of data
            X, Y, start = batcher(start)

            # Do one step of gradient descent.
            #print(f"before model step: {datetime.now().time()}")
            params, state = model.update(params, state, X, Y)
            #print(f"after model step: {datetime.now().time()}")

        # Score the model along the time axis
        #print(f"before computing score: {datetime.now().time()}")
        score_train[k,ep] = model.score(X, Y, score_type="log-likelihood")
        #print(f"after computing score: {datetime.now().time()}")
        #score_test[k,ep] = model.score(X_test, count_test.squeeze(), score_type="log-likelihood")
        print(f"epoch {ep}  compeleted: {datetime.now().time()}")
        print(f"K: {k}, Ep: {ep}, train ll: {score_train[k,ep]}, test ll: {score_test[k,ep]}")

      # model output
    weights.append(model.coef_.reshape(n_fun,-1))
    filters.append(np.matmul(basis_kernels, model.coef_.reshape(n_fun,-1)))
    intercepts.append(model.intercept_)

results_dict = {"weights": weights, "filters": filters, "intercept": intercepts, "type": spike_times_quiet["EI"],
                "time": time, "basis_kernels": basis_kernels, "train_ll": score_train, "test_ll": score_test}

np.save(f"/mnt/home/amedvedeva/ceph/songbird_output/results_n{args.Neuron}.npy", results_dict)




